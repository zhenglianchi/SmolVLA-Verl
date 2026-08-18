#!/usr/bin/env python
"""Local remote trajectory collector v2 (single-episode mode, 16-instance parallel).

Thin client: runs ONE LIBERO episode; per control chunk POSTs the observation to
the remote SmolVLA serving endpoint (/predict) and executes returned actions;
reports outcome via /finish. The training loop launches 16 such instances in
parallel (4 groups x 4 rollouts sharing task+seed) for GRPO grouping.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request

import numpy as np
import torch

SUITE_MAX_STEPS = {"libero_spatial": 280, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def _post(url: str, payload: dict, timeout=120) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _img_b64(img: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(img, dtype=np.uint8).tobytes()).decode()


def _conv(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _conv(v) for k, v in x.items()}
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--rollout-n", type=int, default=4, help="episodes in this group (same scene)")
    ap.add_argument("--group-id", default="")
    ap.add_argument("--session-id", default="")
    ap.add_argument("--eta", type=float, default=0.05,
                    help="SDE noise; keep small so the trained policy transfers to deterministic ODE eval")
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--action-steps", type=int, default=5)
    ap.add_argument("--init-state-id", type=int, default=0,
                    help="LIBERO initial-state index shared by every episode in the group (reset-matched GRPO)")
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()

    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env

    env_cfg = LiberoEnv(
        task=args.suite, task_ids=[args.task_id],
        camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"},
    )
    envs = make_env(env_cfg, n_envs=1)
    env = envs[args.suite][args.task_id]
    # underlying LiberoEnv (VectorEnv wrapper: env.envs[0].unwrapped)
    base_env = env.envs[0].unwrapped
    max_steps = min(SUITE_MAX_STEPS[args.suite], args.max_steps)
    predict_url = args.server.rstrip("/") + "/predict"
    finish_url = args.server.rstrip("/") + "/finish"

    def run_one(episode_id: int, seed: int):
        # vary the policy-noise seed per episode (the scene is pinned below), so
        # the four rollouts of a group differ only through SDE noise
        torch.manual_seed(seed + episode_id * 7919)
        # lerobot LiberoEnv advances init_state_id on every reset; pin it so all
        # episodes of a group restore the SAME initial state (reset-matched)
        base_env.init_state_id = args.init_state_id
        obs, _ = env.reset(seed=seed)
        task_desc = list(env.call("task_description"))[0]
        total = 0
        success = False
        chunk_id = 0
        executed_steps = {}
        while total < max_steps and not success:
            payload = {
                "session_id": args.session_id,
                "episode_id": episode_id,
                "chunk_id": chunk_id,
                "group_id": args.group_id,
                "pixels": {k: _img_b64(v) for k, v in obs["pixels"].items()},
                "robot_state": _conv(obs["robot_state"]),
                "task": task_desc,
                "eta": args.eta,
            }
            r = _post(predict_url, payload)
            acts = np.array(r["actions"], dtype=np.float32).reshape(-1, 7)
            n_exec = 0
            for a in acts:
                obs, reward, terminated, truncated, info = env.step(a.reshape(1, 7))
                n_exec += 1
                total += 1
                final_info = info.get("final_info") if isinstance(info, dict) else None
                if final_info is not None and final_info[0] is not None:
                    success = bool(final_info[0].get("is_success", False))
                else:
                    success = bool(info.get("is_success", False)) if isinstance(info, dict) else False
                if success or bool(np.asarray(truncated).any()):
                    break
            executed_steps[chunk_id] = n_exec
            chunk_id += 1
        _post(finish_url, {"session_id": args.session_id, "episode_id": episode_id,
                           "group_id": args.group_id, "success": success,
                           "executed_steps": executed_steps})
        print(f"[collect-remote] sid={args.session_id} gid={args.group_id} task={args.task_id} "
              f"ep{episode_id} success={success} steps={total}", flush=True)

    group_seed = args.seed
    for i in range(args.rollout_n):
        run_one(i, group_seed)  # same initial scene for the group


if __name__ == "__main__":
    main()
