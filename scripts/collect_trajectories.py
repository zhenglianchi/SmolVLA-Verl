#!/usr/bin/env python
"""Local trajectory collector v2 with per-stage debug timings."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE / "src" / "smolvla_verl" / "models" / "smolvla"))
import grpo  # noqa: E402
import sde_sampling  # noqa: E402

SUITE_MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
_t0 = time.time()


def log(msg):
    print(f"[{time.time()-_t0:7.1f}s] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/root/vla_libero/models/smolvla_libero")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--rollout-n", type=int, default=4)
    ap.add_argument("--groups", type=int, default=1)
    ap.add_argument("--eta", type=float, default=0.05,
                    help="SDE noise; keep small so the trained policy transfers to deterministic ODE eval")
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--action-steps", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--init-state-id", type=int, default=0,
                    help="LIBERO initial-state index shared by every episode in the group (reset-matched GRPO)")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", default="work/trajectories/traj.pkl")
    args = ap.parse_args()

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.envs.utils import preprocess_observation
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors

    log("loading policy...")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    log("policy loaded, setting chunk_size")
    policy.config.chunk_size = args.chunk_size
    policy.eval()
    log("building pre/post processors...")
    _rename_map = {
        "observation.images.camera1": "observation.images.image",
        "observation.images.camera2": "observation.images.image2",
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"rename_observations_processor": {"rename_map": _rename_map}},
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"moving policy to {device}...")
    policy.to(device)
    log(f"model ready; trainable? action_dim={int(policy.config.output_features['action'].shape[0])}")

    log("building env...")
    env_cfg = LiberoEnv(task=args.suite, task_ids=[args.task_id],
                        camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"})
    envs = make_env(env_cfg, n_envs=1)
    env = envs[args.suite][0]
    # underlying LiberoEnv (VectorEnv wrapper: env.envs[0].unwrapped)
    base_env = env.envs[0].unwrapped
    env_pre, _ = make_env_pre_post_processors(env_cfg, policy.config)
    log("env ready")

    action_dim = int(policy.config.output_features["action"].shape[0])
    max_steps = min(SUITE_MAX_STEPS[args.suite], args.max_steps)

    def collect_episode(seed: int, policy_seed: int):
        torch.manual_seed(policy_seed)
        # pin the initial state so every episode in a group is reset-matched
        base_env.init_state_id = args.init_state_id
        obs, _ = env.reset(seed=seed)
        chunks = []
        total = 0
        success = False
        while total < max_steps and not success:
            po = preprocess_observation(obs)
            po["task"] = list(env.call("task_description"))
            po = env_pre(po)
            batch = preprocessor(po)
            with torch.no_grad(), sde_sampling.rollout_autocast(device):
                pm, pc = sde_sampling.prepare_policy_prefix(policy, batch)
                traj = sde_sampling.sample_sde_chunk(
                    policy.model, pm, pc, action_dim=action_dim, eta=args.eta
                )
            actions = traj.actions[:, :, :action_dim]
            actions = postprocessor(actions)
            valid = torch.zeros((1, args.chunk_size), dtype=torch.bool)
            for pos in range(min(args.action_steps, args.chunk_size)):
                a = actions[:, pos].detach().cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(a)
                valid[0, pos] = True
                total += 1
                final_info = info.get("final_info") if isinstance(info, dict) else None
                if final_info is not None and final_info[0] is not None:
                    success = bool(final_info[0].get("is_success", False))
                else:
                    success = bool(info.get("is_success", False)) if isinstance(info, dict) else False
                if success or bool(np.asarray(truncated).any()):
                    break
            # full precision: rescoring must reproduce the collection path
            batch_cpu = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            chunks.append((batch_cpu, traj.to("cpu"), valid.cpu()))
            if total % 40 == 0:
                log(f"ep(seed={seed}) step {total}/{max_steps} chunks={len(chunks)}")
        return {"success": success, "steps": total, "chunks": chunks}

    episodes = []
    for g in range(args.groups):
        group_seed = args.seed + g * 1000
        log(f"=== group {g}: collecting {args.rollout_n} episodes (seed base {group_seed}) ===")
        eps = [collect_episode(group_seed, group_seed + i * 7919) for i in range(args.rollout_n)]
        rewards = torch.tensor([[float(e["success"]) for e in eps]], dtype=torch.float32)
        advs = grpo.compute_group_advantages(rewards)[0].tolist()
        for e, adv in zip(eps, advs):
            e["group_seed"] = group_seed
            e["advantage"] = float(adv)
        episodes.extend(eps)
        n_ok = sum(1 for e in eps if e["success"])
        log(f"group {g} done: success={n_ok}/{len(eps)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"episodes": episodes, "args": vars(args)}, f)
    log(f"saved {len(episodes)} episodes -> {out}")


if __name__ == "__main__":
    main()
