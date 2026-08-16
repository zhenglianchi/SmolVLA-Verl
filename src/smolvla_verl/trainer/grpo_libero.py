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
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from multiprocessing import Pool, set_start_method

# keep the worker imports light
from smolvla_verl.models.smolvla import SmolVLATrainableModel
from smolvla_verl.models.smolvla.grpo import compute_group_advantages, grpo_loss

SUITE_MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
TASK_DESC = "place the bowl on the plate"

# --------------------------------------------------------------------------- #
# rollout worker (process): own env + own frozen model copy
# --------------------------------------------------------------------------- #
class RolloutWorker:
    def __init__(self, checkpoint, suite, task_id, env_cfg, eta, max_steps, action_dim, chunk_size, seed_base):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        self.eta = eta
        self.max_steps = max_steps
        self.action_dim = action_dim
        self.seed_base = seed_base

        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import preprocess_observation
        from verl_vla.models import build_vla_model
        from verl_vla.workers.config.model import VLAModelConfig

        self.preprocess_observation = preprocess_observation
        self._env_cfg = env_cfg
        self._LiberoEnv = LiberoEnv
        self._make_env = make_env
        self._make_env_pre_post_processors = make_env_pre_post_processors

        cfg = VLAModelConfig(path=checkpoint, use_shm=False)
        self.model = build_vla_model(cfg, torch_dtype=torch.bfloat16).to("cuda")
        self.model.policy.config.chunk_size = chunk_size
        self.model.eval()
        self.envs = None
        self.env = None
        self.env_preprocessor = None

    def _ensure_env(self):
        if self.env is None:
            from lerobot.envs.configs import LiberoEnv
            env_cfg = LiberoEnv(
                task=self._env_cfg["task"],
                task_ids=[self._env_cfg["task_id"]],
                camera_name_mapping=self._env_cfg["camera_name_mapping"],
            )
            envs = self._make_env(env_cfg, n_envs=1)
            self.env = envs[env_cfg.task][0]
            self.env_preprocessor, _ = self._make_env_pre_post_processors(env_cfg, self.model.policy.config)

    def collect_episode(self, seed: int):
        self._ensure_env()
        from verl import DataProto
        torch.manual_seed(seed)
        obs, _ = self.env.reset(seed=seed)
        chunks = []
        total_steps = 0
        success = False
        max_steps = self.max_steps
        while total_steps < max_steps and not success:
            policy_obs = self.preprocess_observation(obs)
            policy_obs["task"] = list(self.env.call("task_description"))
            policy_obs = self.env_preprocessor(policy_obs)
            data = {k: v for k, v in policy_obs.items() if k.startswith("observation.")}
            data["task"] = np.asarray([TASK_DESC])
            dp = DataProto.from_single_dict(data)
            with self.model.rollout_context():
                traj = self.model.sample_sde_chunk(dp, eta=self.eta)
            actions = traj.actions[:, :, :self.action_dim]
            horizon = min(self.model.policy.config.n_action_steps, max_steps - total_steps)
            valid_positions = torch.zeros((1, self.model.policy.config.chunk_size), dtype=torch.bool)
            for position in range(horizon):
                a = actions[:, position].detach().cpu().numpy()
                obs, reward, terminated, truncated, info = self.env.step(a)
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

    def collect_group(self, group_seed: int):
        """One reset-matched group: G episodes from the same initial scene."""
        episodes = []
        for i in range(self._group_size):
            episodes.append(self.collect_episode(self.seed_base + group_seed * 97 + i))
        rewards = torch.tensor([[float(e["success"]) for e in episodes]], dtype=torch.float32)
        advs = compute_group_advantages(rewards)[0].tolist()
        for e, adv in zip(episodes, advs):
            e["advantage"] = float(adv)
        return episodes


def _worker_init(checkpoint, suite, task_id, env_cfg, eta, max_steps, action_dim, chunk_size, seed_base, group_size):
    global _WORKER
    _WORKER = RolloutWorker(checkpoint, suite, task_id, env_cfg, eta, max_steps, action_dim, chunk_size, seed_base)
    _WORKER._group_size = group_size


def _collect_one_group(args):
    return _WORKER.collect_group(args)


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
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip-epsilon", type=float, default=0.2)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--save-dir", default="/root/runs/smolvla_grpo")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-after", action="store_true", help="evaluate after each round (uses scripts/eval)")
    args = ap.parse_args()

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
        "task_id": int(args.task_ids.split(",")[0]),
        "camera_name_mapping": {"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"},
    }
    action_dim = int(model.policy.config.output_features["action"].shape[0])

    set_start_method("spawn", force=True)
    pool = Pool(
        args.num_runners,
        initializer=_worker_init,
        initargs=(args.checkpoint, args.suite, env_cfg["task_id"], env_cfg, args.eta,
                  min(SUITE_MAX_STEPS[args.suite], args.max_steps), action_dim, args.chunk_size, args.seed, args.rollout_n),
    )

    for rnd in range(start_round, args.rounds + 1):
        t_r = time.time()
        print(f"\n=== round {rnd}/{args.rounds} ===", flush=True)
        group_seeds = [rnd * 1000 + g for g in range(args.groups_per_round)]
        results = pool.map(_collect_one_group, group_seeds)
        episodes = [e for group in results for e in group]
        n_ok = sum(1 for e in episodes if e["success"])
        print(f"[rollout] episodes={len(episodes)} success={n_ok}/{len(episodes)} "
              f"({100.0*n_ok/len(episodes):.1f}%) time={time.time()-t_r:.0f}s", flush=True)

        # rescore + GRPO loss on main GPU
        ref.to("cuda")
        total_loss = 0.0
        ratio_acc = 0.0
        nchunks = 0
        for e in episodes:
            adv = e["advantage"]
            for dp, traj, valid_positions in e["chunks"]:
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
                adv_t = torch.tensor([adv], device=logp.device, dtype=logp.dtype)
                loss, metrics = grpo_loss(
                    logp, old_per_step, ref_per_step.to(logp.device), adv_t,
                    clip_epsilon=args.clip_epsilon, kl_beta=args.kl_beta,
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

    pool.close()
    pool.join()
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()