#!/usr/bin/env python
"""Online FlowGRPO training for SmolVLA on LIBERO.

Designed for the SmolVLA-Verl platform:
  * local/edge: only trajectory collection + upload (see scripts/collect_*.sh)
  * server: this script runs GRPO training with parallel rollout runners

Key options:
  --rollout-n G        GRPO group size (rollouts sharing one initial state)
  --rounds R           total RL training rounds (updates)
  --num-runners N      parallel rollout worker processes (each: env + model copy)
  --save-dir DIR       single weights folder, overwritten every round (resume-safe)
  --resume             reload latest weights from save-dir if present

Fixes over the initial implementation:
  * the environment's real task description is used for every episode (no
    hardcoded instruction), and tasks rotate across rounds/groups;
  * every episode inside a GRPO group restores the SAME initial state
    (``LiberoEnv.init_state_id``) so the group baseline is reset-matched;
  * rollout workers are recreated every round with the latest saved weights
    (on-policy rollout), instead of being pinned to the base checkpoint;
  * ``pool.map`` runs all episodes of a round in parallel (not one at a time);
  * collapsed groups (all-success / all-failure) are skipped entirely;
  * chunks are weighted episode-balanced and a reward-to-go discount
    (--chunk-discount) concentrates credit near the terminal chunk.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from multiprocessing import Pool, set_start_method

from smolvla_verl.models.smolvla import SmolVLATrainableModel
from smolvla_verl.models.smolvla.grpo import compute_group_advantages, grpo_loss

SUITE_MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
RATIO_TOLERANCE = 0.05


# --------------------------------------------------------------------------- #
# rollout worker (process): own env + own frozen model copy
# --------------------------------------------------------------------------- #
class RolloutWorker:
    def __init__(self, checkpoint, suite, env_cfg, eta, max_steps, action_steps, action_dim, chunk_size):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        self.eta = eta
        self.max_steps = max_steps
        self.action_dim = action_dim
        self._action_steps = action_steps
        self._suite = suite
        self._env_cfg = env_cfg

        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        from verl_vla.models import build_vla_model
        from verl_vla.workers.config.model import VLAModelConfig

        self.preprocess_observation = preprocess_observation
        self._LiberoEnv = LiberoEnv
        self._make_env = make_env
        self._make_env_pre_post_processors = make_env_pre_post_processors

        cfg = VLAModelConfig(path=checkpoint, use_shm=False)
        self.model = build_vla_model(cfg, torch_dtype=torch.bfloat16).to("cuda")
        self.model.policy.config.chunk_size = chunk_size
        self.model.eval()
        self._envs = {}
        self._base_envs = {}
        self._env_preprocessors = {}

    def _ensure_env(self, task_id):
        if task_id not in self._envs:
            env_cfg = self._LiberoEnv(
                task=self._suite,
                task_ids=[task_id],
                camera_name_mapping=self._env_cfg["camera_name_mapping"],
            )
            envs = self._make_env(env_cfg, n_envs=1)
            self._envs[task_id] = envs[env_cfg.task][0]
            self._base_envs[task_id] = envs[env_cfg.task][0].envs[0].unwrapped
            self._env_preprocessors[task_id], _ = self._make_env_pre_post_processors(
                env_cfg, self.model.policy.config
            )
        return self._envs[task_id], self._env_preprocessors[task_id]

    def collect_episode(self, task_id, init_state_id, env_seed, policy_seed):
        from verl import DataProto

        env, env_preprocessor = self._ensure_env(task_id)
        base_env = self._base_envs[task_id]
        torch.manual_seed(policy_seed)
        # lerobot LiberoEnv advances init_state_id on every reset; pin it so all
        # episodes of a group restore the SAME initial state (reset-matched).
        base_env.init_state_id = init_state_id
        obs, _ = env.reset(seed=env_seed)
        task_desc = np.asarray(list(env.call("task_description")))
        chunks = []
        total_steps = 0
        success = False
        max_steps = self.max_steps
        while total_steps < max_steps and not success:
            policy_obs = self.preprocess_observation(obs)
            policy_obs["task"] = task_desc
            policy_obs = env_preprocessor(policy_obs)
            data = {k: v for k, v in policy_obs.items() if k.startswith("observation.")}
            data["task"] = task_desc
            dp = DataProto.from_single_dict(data)
            with self.model.rollout_context():
                traj = self.model.sample_sde_chunk(dp, eta=self.eta)
            actions = traj.actions[:, :, : self.action_dim]
            # unnormalize (MEAN_STD) before feeding the environment
            actions = self.model.postprocessor(actions)
            horizon = min(self._action_steps, max_steps - total_steps)
            valid_positions = torch.zeros((1, self.model.policy.config.chunk_size), dtype=torch.bool)
            for position in range(horizon):
                a = actions[:, position].detach().cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(a)
                valid_positions[0, position] = True
                total_steps += 1
                final_info = info.get("final_info")
                if final_info is not None and final_info[0] is not None:
                    success = bool(final_info[0].get("is_success", False))
                else:
                    success = bool(info.get("is_success", False))
                if success or bool(np.asarray(truncated).any()):
                    break
            chunks.append((dp.to("cpu"), traj.to("cpu"), valid_positions.cpu()))
        return {"success": success, "steps": total_steps, "chunks": chunks}

    def collect_episode_task(self, task_id, init_state_id, env_seed, policy_seed):
        """One episode within a reset-matched group."""
        return self.collect_episode(task_id, init_state_id, env_seed, policy_seed)


def _worker_init(checkpoint, suite, env_cfg, eta, max_steps, action_steps, action_dim, chunk_size):
    global _WORKER
    _WORKER = RolloutWorker(
        checkpoint, suite, env_cfg, eta, max_steps, action_steps, action_dim, chunk_size
    )


def _collect_one_episode(args):
    task_id, init_state_id, env_seed, policy_seed = args
    return _WORKER.collect_episode_task(task_id, init_state_id, env_seed, policy_seed)


# --------------------------------------------------------------------------- #
# main trainer
# --------------------------------------------------------------------------- #
def build_trainable(checkpoint, chunk_size):
    from verl_vla.models import build_vla_model
    from verl_vla.workers.config.model import VLAModelConfig

    cfg = VLAModelConfig(path=checkpoint, use_shm=False)
    model = build_vla_model(cfg, torch_dtype=torch.bfloat16).to("cuda")
    model.policy.config.chunk_size = chunk_size
    model.eval()
    return model


def build_reference(checkpoint, chunk_size):
    from verl_vla.models import build_vla_model
    from verl_vla.workers.config.model import VLAModelConfig

    cfg = VLAModelConfig(path=checkpoint, use_shm=False)
    ref = build_vla_model(cfg, torch_dtype=torch.bfloat16).to("cuda")
    ref.policy.config.chunk_size = chunk_size
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/root/models/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", default="0")
    ap.add_argument("--rollout-n", type=int, default=4, help="GRPO group size")
    ap.add_argument("--groups-per-round", type=int, default=4, help="groups collected per round")
    ap.add_argument("--rounds", type=int, default=10, help="RL training rounds")
    ap.add_argument("--num-runners", type=int, default=4, help="parallel rollout worker processes")
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--action-steps", type=int, default=1, help="chunk actions executed per sample (1 = FlowVLA-RL semantics; >1 cuts sampling/rescoring cost)")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip-epsilon", type=float, default=0.2)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--chunk-discount", type=float, default=0.99,
                    help="reward-to-go discount across chunks of one episode, (0,1]")
    ap.add_argument("--init-state-count", type=int, default=10,
                    help="rotate training initial states across 0..N-1 to match eval coverage")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--save-dir", default="/root/runs/smolvla_grpo")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-after", action="store_true", help="evaluate after each round (uses scripts/eval)")
    args = ap.parse_args()
    if not 0.0 < args.chunk_discount <= 1.0:
        ap.error("--chunk-discount must lie in (0, 1]")
    if args.init_state_count <= 0:
        ap.error("--init-state-count must be positive")
    task_ids = [int(t.strip()) for t in args.task_ids.split(",") if t.strip()]
    if not task_ids:
        ap.error("--task-ids must contain at least one task id")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    start_round = 0
    if args.resume and (save_dir / "model.safetensors").exists():
        # weights already in save_dir; trainer will load them as the base for this run
        start_round = 1
        print(f"[resume] weights found in {save_dir}, starting at round {start_round}")

    t0 = time.time()
    model = build_trainable(args.checkpoint, args.chunk_size)
    ref = build_reference(args.checkpoint, args.chunk_size)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"[setup] trainable={sum(p.numel() for p in trainable)/1e6:.2f}M "
          f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB time={time.time()-t0:.1f}s")

    env_cfg = {
        "task": args.suite,
        "camera_name_mapping": {"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"},
    }
    action_dim = int(model.policy.config.output_features["action"].shape[0])
    max_steps = min(SUITE_MAX_STEPS[args.suite], args.max_steps)

    def _current_checkpoint() -> str:
        return str(save_dir) if (save_dir / "model.safetensors").exists() else args.checkpoint

    if args.num_runners > 1:
        set_start_method("spawn", force=True)

    pool = None

    def _make_pool(checkpoint: str):
        nonlocal pool
        if pool is not None:
            pool.close()
            pool.join()
            pool = None
        if args.num_runners > 1:
            pool = Pool(
                args.num_runners,
                initializer=_worker_init,
                initargs=(checkpoint, args.suite, env_cfg, args.eta,
                          max_steps, args.action_steps, action_dim, args.chunk_size),
            )

    def _run_tasks(tasks):
        if pool is not None:
            return pool.map(_collect_one_episode, tasks)
        _worker_init(_current_checkpoint(), args.suite, env_cfg, args.eta,
                     max_steps, args.action_steps, action_dim, args.chunk_size)
        return [_collect_one_episode(task) for task in tasks]

    for rnd in range(start_round, args.rounds):
        t_r = time.time()
        print(f"\n=== round {rnd}/{args.rounds} ===", flush=True)
        # recreate the pool with the latest weights: on-policy rollout
        _make_pool(_current_checkpoint())

        # tasks rotate across tasks AND initial states; every episode of a group
        # shares (task, init_state_id, env_seed) and only the policy noise seed
        # differs, so the GRPO group is reset-matched.
        tasks = []
        for g in range(args.groups_per_round):
            task_id = task_ids[(rnd * args.groups_per_round + g) % len(task_ids)]
            init_state_id = (rnd + g) % args.init_state_count
            env_seed = args.seed + rnd * 1000 + g
            for i in range(args.rollout_n):
                policy_seed = args.seed + 100_000 + rnd * 1000 + g * args.rollout_n + i
                tasks.append((task_id, init_state_id, env_seed, policy_seed))
        results = _run_tasks(tasks)

        # regroup by group and compute reset-matched advantages (mixed groups only)
        grouped = {g: [] for g in range(args.groups_per_round)}
        for g in range(args.groups_per_round):
            grouped[g] = results[g * args.rollout_n : (g + 1) * args.rollout_n]
        episodes = []
        for g, eps in grouped.items():
            successes = [float(e["success"]) for e in eps]
            if all(s == successes[0] for s in successes):
                # collapsed group: no within-group signal; skip entirely
                continue
            rewards = torch.tensor([successes], dtype=torch.float32)
            advs = compute_group_advantages(rewards)[0].tolist()
            for e, adv in zip(eps, advs):
                e["advantage"] = float(adv)
            episodes.extend(eps)
        n_ok = sum(1 for e in episodes if e["success"])
        print(f"[rollout] mixed episodes={len(episodes)} success={n_ok}/{len(episodes)} "
              f"({100.0 * n_ok / max(len(episodes), 1):.1f}%) time={time.time()-t_r:.0f}s", flush=True)
        if not episodes:
            print("[rollout] all groups collapsed; skipping update this round", flush=True)
            model.save_pretrained(str(save_dir))
            if pool is not None:
                pool.close()
                pool.join()
                pool = None
            continue

        # rescore + GRPO loss on main GPU
        ref.to("cuda")
        total_loss = 0.0
        ratio_acc = 0.0
        nchunks = 0
        checked_ratio = False
        n_episodes = max(len(episodes), 1)
        for e in episodes:
            adv = e["advantage"]
            n_chunks = len(e["chunks"])
            ep_weight = 1.0 / (n_episodes * n_chunks)
            for cpos, (dp, traj, valid_positions) in enumerate(e["chunks"]):
                dp = dp.to("cuda")
                traj = traj.to("cuda")
                with torch.no_grad(), model.rollout_context():
                    ref_per_step = ref.flow_log_prob(dp, traj, valid_positions=valid_positions).unsqueeze(1).cpu()
                with model.rollout_context():
                    logp = model.flow_log_prob(dp, traj, valid_positions=valid_positions).unsqueeze(1)
                    old_per_step = (
                        traj.element_log_probs * traj.valid_action_mask[:, None].float()
                        * valid_positions.to(traj.states.device)[None, ..., None].float()
                    ).sum(dim=(-1, -2)).unsqueeze(1)
                chunk_adv = adv * (args.chunk_discount ** (n_chunks - 1 - cpos))
                adv_t = torch.tensor([chunk_adv], device=logp.device, dtype=logp.dtype)
                loss, metrics = grpo_loss(
                    logp, old_per_step, ref_per_step.to(logp.device), adv_t,
                    clip_epsilon=args.clip_epsilon, kl_beta=args.kl_beta,
                    sample_weights=torch.tensor([ep_weight], device=logp.device, dtype=logp.dtype),
                )
                if not checked_ratio:
                    checked_ratio = True
                    drift = abs(float(metrics["ratio_mean"].cpu()) - 1.0)
                    if drift > RATIO_TOLERANCE:
                        raise RuntimeError(
                            f"first-forward ratio_mean={float(metrics['ratio_mean'].cpu()):.4f} "
                            f"drifts {drift:.4f} from 1.0; collection and rescoring are on "
                            "different numeric paths (both sides must use rollout_autocast)."
                        )
                loss.backward()
                total_loss += float(loss.detach().cpu())
                ratio_acc += float(metrics["ratio_mean"].cpu())
                nchunks += 1
        ref.to("cpu")
        torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        optim.zero_grad()
        print(f"[train] chunks={nchunks} loss={total_loss:.4f} ratio_mean={ratio_acc/max(nchunks,1):.4f} "
              f"time={time.time()-t_r:.0f}s", flush=True)

        # save single weights folder (overwrite)
        model.save_pretrained(str(save_dir))
        print(f"[save] weights -> {save_dir} (overwrite)", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
            pool = None

    if pool is not None:
        pool.close()
        pool.join()
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
