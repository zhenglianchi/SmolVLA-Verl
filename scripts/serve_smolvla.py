#!/usr/bin/env python
"""SmolVLA action serving endpoint (server side), Gateway-style recording.

Clients (local env runners) POST observations -> receive actions + log-probs.
The server records each served chunk (obs + SDE states + log-probs); clients
report episode outcome via /finish, which also computes group advantages.
GRPO training (grpo_offline) then consumes the recorded sessions.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE / "src" / "smolvla_verl" / "models" / "smolvla"))
import sde_sampling  # noqa: E402

app = FastAPI()
POLICY = None
PREPROCESSOR = None
POSTPROCESSOR = None
ENV_PRE = None
CHECKPOINT = ""
SAVE_DIR = ""
CHUNK_SIZE = 10
ACTION_STEPS = 5
ACTION_DIM = 7

# session recording: session_id -> {episode_id -> {chunk_id -> chunk_data}}
SESSIONS: dict = {}
SESSIONS_LOCK = threading.Lock()
# group advantages bookkeeping: group_id -> {episode_id -> success}
GROUP_RESULTS: dict = {}


class PredictRequest(BaseModel):
    session_id: str
    episode_id: int
    chunk_id: int
    group_id: str | None = None
    pixels: dict[str, str]
    robot_state: dict
    task: str
    eta: float = 0.1


class FinishRequest(BaseModel):
    session_id: str
    episode_id: int
    group_id: str
    success: bool
    # chunk_id -> number of actions actually executed in that chunk; lets the
    # server mask planned-but-unexecuted suffix actions of the terminal chunk
    executed_steps: dict[str, int] | None = None


def _decode_img(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    px = arr.size // 3
    side = int(round(px ** 0.5))
    return arr.reshape(side, px // side, 3)




def _to_tensors(d):
    if isinstance(d, dict):
        return {k: _to_tensors(v) for k, v in d.items()}
    if isinstance(d, list):
        return torch.tensor(d, dtype=torch.float32)
    return d

def _load(checkpoint, chunk_size):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors

    global POLICY, PREPROCESSOR, POSTPROCESSOR, ENV_PRE, CHECKPOINT, CHUNK_SIZE
    POLICY = SmolVLAPolicy.from_pretrained(checkpoint)
    POLICY.config.chunk_size = chunk_size
    POLICY.eval()
    POLICY.to("cuda")
    _rm = {
        "observation.images.camera1": "observation.images.image",
        "observation.images.camera2": "observation.images.image2",
    }
    PREPROCESSOR, POSTPROCESSOR = make_pre_post_processors(
        POLICY.config, pretrained_path=checkpoint,
        preprocessor_overrides={"rename_observations_processor": {"rename_map": _rm}},
    )
    env_cfg = LiberoEnv(
        task="libero_spatial", task_ids=[0],
        camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"},
    )
    ENV_PRE, _ = make_env_pre_post_processors(env_cfg, POLICY.config)
    CHECKPOINT = checkpoint
    CHUNK_SIZE = chunk_size


def _tob64(t: torch.Tensor) -> str:
    return base64.b64encode(t.detach().cpu().contiguous().numpy().tobytes()).decode()


def _fromb64(b64: str, shape, dtype=np.float32) -> torch.Tensor:
    arr = np.frombuffer(base64.b64decode(b64), dtype=dtype).reshape(shape)
    return torch.from_numpy(arr)


@app.post("/predict")
def predict(req: PredictRequest):
    from lerobot.envs.utils import preprocess_observation

    pixels = {k: _decode_img(v) for k, v in req.pixels.items()}
    obs = {"pixels": pixels, "robot_state": req.robot_state}
    po = preprocess_observation(obs)
    po["task"] = [req.task]
    if "observation.robot_state" in po:
        po["observation.robot_state"] = _to_tensors(po["observation.robot_state"])
    po = ENV_PRE(po)
    batch = PREPROCESSOR(po)
    with torch.no_grad(), sde_sampling.rollout_autocast("cuda"):
        pm, pc = sde_sampling.prepare_policy_prefix(POLICY, batch)
        traj = sde_sampling.sample_sde_chunk(
            POLICY.model, pm, pc, action_dim=ACTION_DIM, eta=req.eta
        )
    actions = traj.actions[:, :, :ACTION_DIM]
    actions = POSTPROCESSOR(actions)
    n_exec = min(ACTION_STEPS, CHUNK_SIZE)

    # record the chunk (obs batch CPU + SDE trajectory) for later training
    # store images at full precision: rescoring must reproduce the collection
    # numeric path bit-for-bit (half-precision storage silently breaks ratio=1)
    batch_cpu = {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    with SESSIONS_LOCK:
        SESSIONS.setdefault(req.session_id, {})[req.episode_id] = SESSIONS.get(req.session_id, {}).get(
            req.episode_id, {"group_id": req.group_id, "chunks": {}}
        )
        SESSIONS[req.session_id][req.episode_id]["chunks"][req.chunk_id] = {
            "batch": batch_cpu,
            "states": _tob64(traj.states),
            "element_log_probs": _tob64(traj.element_log_probs),
            "valid_action_mask": _tob64(traj.valid_action_mask),
            "valid_positions": _tob64(torch.zeros((1, CHUNK_SIZE), dtype=torch.bool).index_fill(1, torch.arange(n_exec), True)),
            "n_exec": n_exec,
            "eta": req.eta,
            "success": False,
        }
    return {
        "actions": actions[0, :n_exec].detach().cpu().numpy().tolist(),
        "chunk_log_prob": float(traj.log_probs.detach().cpu().sum().item()),
    }


@app.post("/finish")
def finish(req: FinishRequest):
    """Mark episode outcome; when a group completes, compute reset-matched advantages."""
    with SESSIONS_LOCK:
        sess = SESSIONS.get(req.session_id)
        if sess is None:
            return {"status": "session not found"}
        ep = sess.get(req.episode_id)
        if ep is None:
            return {"status": "episode not found"}
        if req.executed_steps:
            for cid_str, n_exec in req.executed_steps.items():
                ch = ep["chunks"].get(int(cid_str))
                if ch is None:
                    continue
                n_exec = min(int(n_exec), CHUNK_SIZE)
                if n_exec < ch["n_exec"]:
                    # mask the planned-but-never-executed suffix of the terminal chunk
                    vp = np.frombuffer(base64.b64decode(ch["valid_positions"]), dtype=np.bool_).copy()
                    vp = vp.reshape(1, CHUNK_SIZE)
                    vp[0, n_exec:] = False
                    ch["valid_positions"] = _tob64(torch.from_numpy(vp))
                    ch["n_exec"] = n_exec
        ep["success"] = req.success
        ep["steps"] = len(ep["chunks"]) * ACTION_STEPS
        GROUP_RESULTS.setdefault(req.group_id, {})[req.episode_id] = req.success
    return {"status": "ok"}


@app.post("/train")
def train(lr: float = 1e-6, steps: int = 1, clip_epsilon: float = 0.2, kl_beta: float = 0.01,
          batch_size: int = 1, chunk_discount: float = 0.99):
    """Run one or more offline GRPO steps on recorded sessions, then save weights."""
    from smolvla_verl.trainer.grpo_offline import train_from_sessions

    result = train_from_sessions(SESSIONS, GROUP_RESULTS, POLICY, PREPROCESSOR, POSTPROCESSOR, SAVE_DIR, CHECKPOINT, CHUNK_SIZE,
                                 lr=lr, clip_epsilon=clip_epsilon, kl_beta=kl_beta, steps=steps, batch_size=batch_size,
                                 chunk_discount=chunk_discount)
    return result


@app.post("/reload")
def reload_weights():
    global POLICY
    if not SAVE_DIR or not Path(SAVE_DIR).joinpath("model.safetensors").exists():
        return {"status": "no weights yet"}
    POLICY.cpu()
    torch.cuda.empty_cache()
    _load(SAVE_DIR, CHUNK_SIZE)
    return {"status": "reloaded from " + SAVE_DIR}


@app.post("/clear")
def clear_sessions():
    with SESSIONS_LOCK:
        SESSIONS.clear()
        GROUP_RESULTS.clear()
    return {"status": "cleared"}


@app.get("/stats")
def stats():
    out = {}
    for sid, eps in SESSIONS.items():
        out[sid] = {str(eid): {"success": ep.get("success", False), "steps": ep.get("steps", 0),
                                "chunks": len(ep.get("chunks", {}))}
                    for eid, ep in eps.items()}
    return {"groups": GROUP_RESULTS, "episodes": out}


@app.post("/dump")
def dump_sessions():
    """Debug: pickle the recorded sessions + group results to disk."""
    import pickle

    out_path = Path(SAVE_DIR) / "session_dump.pkl"
    with SESSIONS_LOCK:
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "sessions": dict(SESSIONS),
                    "group_results": GROUP_RESULTS,
                    "chunk_size": CHUNK_SIZE,
                    "action_steps": ACTION_STEPS,
                    "action_dim": ACTION_DIM,
                },
                f,
            )
    return {"status": "dumped", "path": str(out_path), "episodes": sum(len(eps) for eps in SESSIONS.values())}


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CHECKPOINT, "sessions": len(SESSIONS)}


def main():
    global SAVE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/home/ubuntu/models/smolvla_libero")
    ap.add_argument("--save-dir", default="/home/ubuntu/runs/smolvla_grpo")
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--action-steps", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    global ACTION_STEPS
    SAVE_DIR = args.save_dir
    ACTION_STEPS = args.action_steps
    _load(args.checkpoint, args.chunk_size)
    print(f"[serve] loaded {args.checkpoint}, listening on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
