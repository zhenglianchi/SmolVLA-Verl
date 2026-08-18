#!/usr/bin/env python
"""Offline FlowGRPO trainer (server side, no environment).

Consumes trajectory files collected locally (uniagent-lighting platform
pattern): rescore stored trajectories with current weights, GRPO loss vs the
stored old log-probs, optimize the action expert, save a single weights folder.
"""
from __future__ import annotations

import base64
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_HERE / "src" / "smolvla_verl" / "models" / "smolvla"))
import grpo  # noqa: E402
import sde_sampling  # noqa: E402

RATIO_TOLERANCE = 0.05


def load_policy(checkpoint, chunk_size, freeze=False):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.config.chunk_size = chunk_size
    policy.eval()
    policy.to("cuda")
    if freeze:
        for p in policy.parameters():
            p.requires_grad_(False)
    return policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default="/home/ubuntu/trajectories/traj.pkl")
    ap.add_argument("--checkpoint", default="/home/ubuntu/models/smolvla_libero")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip-epsilon", type=float, default=0.2)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs-per-round", type=int, default=1)
    ap.add_argument("--save-dir", default="/home/ubuntu/runs/smolvla_grpo")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()

    t0 = time.time()
    with open(args.trajectories, "rb") as f:
        data = pickle.load(f)
    episodes = data["episodes"]
    chunk_size = data["args"].get("chunk_size", 10)
    print(f"[offline] loaded {len(episodes)} episodes from {args.trajectories}", flush=True)

    policy = load_policy(args.checkpoint, chunk_size)
    ref = load_policy(args.checkpoint, chunk_size, freeze=True)
    from lerobot.policies.factory import make_pre_post_processors

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint
    )
    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"[offline] trainable={sum(p.numel() for p in trainable)/1e6:.2f}M "
          f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB time={time.time()-t0:.0f}s", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume and (save_dir / "model.safetensors").exists():
        start = 1
        print(f"[offline] resume: weights exist in {save_dir}", flush=True)

    def rescore(pol, batch_cpu, traj, valid):
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch_cpu.items()}
        pm, pc = sde_sampling.prepare_policy_prefix(pol, batch)
        return sde_sampling.recompute_log_probs(pol.model, pm, pc, traj, valid_positions=valid)

    # collapsed groups (all-same outcome) carry zero advantage and no signal;
    # drop them so the objective is not dominated by the KL-to-base term.
    episodes = [ep for ep in episodes if float(ep.get("advantage", 0.0)) != 0.0]
    if not episodes:
        raise RuntimeError("no mixed-group episodes with non-zero advantage; nothing to train on")
    episode_weights = [1.0 / (len(episodes) * max(len(ep["chunks"]), 1)) for ep in episodes]

    for rnd in range(start, args.rounds):
        t_r = time.time()
        print(f"\n=== offline round {rnd}/{args.rounds} ===", flush=True)
        total_loss = 0.0
        ratio_acc = 0.0
        nchunks = 0
        for epoch in range(args.epochs_per_round):
            for ep, ep_weight in zip(episodes, episode_weights):
                adv = ep["advantage"]
                for batch_cpu, traj, valid in ep["chunks"]:
                    traj = traj.to("cuda")
                    valid = valid.to("cuda")
                    with sde_sampling.rollout_autocast("cuda"):
                        logp = rescore(policy, batch_cpu, traj, valid).unsqueeze(1)  # grad
                        with torch.no_grad():
                            ref_logp = rescore(ref, batch_cpu, traj, valid).unsqueeze(1)
                    old_per_step = (
                        traj.element_log_probs * traj.valid_action_mask[:, None].float()
                        * valid.to(traj.states.device)[None, ..., None].float()
                    ).sum(dim=(-1, -2)).unsqueeze(1)
                    adv_t = torch.tensor([adv], device=logp.device, dtype=logp.dtype)
                    loss, metrics = grpo.grpo_loss(
                        logp, old_per_step, ref_logp, adv_t,
                        clip_epsilon=args.clip_epsilon, kl_beta=args.kl_beta,
                        sample_weights=torch.tensor([ep_weight], device=logp.device, dtype=logp.dtype),
                    )
                    loss.backward()
                    total_loss += float(loss.detach().cpu())
                    ratio_acc += float(metrics["ratio_mean"].cpu())
                    nchunks += 1
                    if rnd == 0 and epoch == 0 and nchunks == 1:
                        drift = abs(float(metrics["ratio_mean"].cpu()) - 1.0)
                        if drift > RATIO_TOLERANCE:
                            raise RuntimeError(
                                f"first-forward ratio_mean={float(metrics['ratio_mean'].cpu()):.4f} "
                                f"drifts {drift:.4f} from 1.0; collection and rescoring are on "
                                "different numeric paths (both sides must use rollout_autocast "
                                "and identical input dtypes)."
                            )
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        optim.zero_grad()
        print(f"[offline] chunks={nchunks} loss={total_loss:.4f} ratio_mean={ratio_acc/max(nchunks,1):.4f} "
              f"time={time.time()-t_r:.0f}s", flush=True)
        policy.save_pretrained(str(save_dir))
    preprocessor.save_pretrained(str(save_dir))
    postprocessor.save_pretrained(str(save_dir))
    print(f"[offline] weights -> {save_dir} (overwrite)", flush=True)
    print("OFFLINE_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()

def _stack_batches(batches, max_lang=None):
    """Stack per-chunk preprocessed batches (each batch_size 1) into one mini-batch.

    Image/state tensors are concatenated along dim 0; language token/mask tensors
    are padded to the longest sequence present in the mini-batch.
    """
    if max_lang is None:
        max_lang = 0
        for b in batches:
            lt = b.get("observation.language.tokens")
            if lt is not None and lt.dim() >= 2:
                max_lang = max(max_lang, lt.shape[1])
    out = {}
    for k in batches[0].keys():
        tensors = [b[k] for b in batches if k in b and isinstance(b[k], torch.Tensor)]
        if not tensors:
            continue
        if k in ("observation.language.tokens", "observation.language.attention_mask"):
            B = len(tensors)
            padded = torch.zeros((B, max_lang), dtype=tensors[0].dtype, device=tensors[0].device)
            for i, t in enumerate(tensors):
                slen = min(t.shape[1], max_lang)
                padded[i, :slen] = t[0, :slen]
            out[k] = padded
        else:
            out[k] = torch.cat(tensors, dim=0)
    return out


def train_from_sessions(sessions, group_results, policy, preprocessor, postprocessor, save_dir, checkpoint, chunk_size,
                        lr=1e-6, clip_epsilon=0.2, kl_beta=0.01, epochs=1, rounds=1,
                        steps=1, batch_size=16, chunk_discount=0.99):
    """Train GRPO on server-recorded sessions (called by /train).

    Numeric-consistency design (verl-style) + STABLE reference anchor:
      - Pass A (no grad, batched): recompute OLD per-step log-probs under the
        round-start policy (for the ratio) AND REFERENCE per-step log-probs
        under a FIXED base policy (SmolVLA-LIBERO) for the KL anchor. Anchoring
        to the fixed base prevents the cumulative policy drift that occurs when
        the reference is the round-start policy.
      - Passes B (grad, batched): recompute NEW log-probs under the (updated)
        policy; ratio = exp(new - old); KL = k3(new, ref_base). ``steps``
        gradient steps total.
    Batching is safe because old/new/ref all share the same batched path.

    Only mixed groups (within-group success differs) carry a learning signal
    and are trained on. Each chunk is weighted episode-balanced (``1/(E*C)``)
    times a reward-to-go discount ``chunk_discount**(C-1-c)`` so that long
    failed episodes no longer dominate the objective and credit concentrates
    near the terminal chunk.
    """
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    torch.manual_seed(20260816)
    # Fixed base reference for the KL anchor (prevents drift)
    ref_checkpoint = "/home/ubuntu/models/smolvla_libero"
    ref = SmolVLAPolicy.from_pretrained(ref_checkpoint)
    ref.config.chunk_size = chunk_size
    ref.eval()
    ref.to("cuda")
    for pp in ref.parameters():
        pp.requires_grad_(False)

    trainable = [pp for pp in policy.model.parameters() if pp.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=lr)

    # assemble episodes with advantages; keep only mixed (reset-matched) groups
    episodes = []
    for sid, eps in sessions.items():
        for ep_id, ep in eps.items():
            episodes.append(ep)
    mixed_groups = set()
    for gid in group_results:
        eps_in_group = [ep for ep in episodes if ep.get("group_id") == gid]
        if len(eps_in_group) >= 2:
            successes = [float(ep.get("success", False)) for ep in eps_in_group]
            if all(s == successes[0] for s in successes):
                # collapsed group: all-same outcome -> zero advantage and no
                # signal; including it would only apply the KL-to-base term.
                continue
            mixed_groups.add(gid)
            rewards = torch.tensor([successes], dtype=torch.float32)
            advs = grpo.compute_group_advantages(rewards)[0].tolist()
            for ep, adv in zip(eps_in_group, advs):
                ep["advantage"] = float(adv)

    selected = [ep for ep in episodes if ep.get("group_id") in mixed_groups and "advantage" in ep]
    n_episodes = len(selected)
    if n_episodes == 0:
        print("[train] no mixed groups in this round; skipping update", flush=True)
        return {"status": "trained", "skipped": "no mixed groups", "chunks": 0}

    # decode all chunks once (hold on GPU); episode-balanced x reward-to-go weights
    num_steps = int(policy.config.num_steps)
    max_action_dim = int(policy.config.max_action_dim)
    items = []
    for ep in selected:
        adv = ep.get("advantage", 0.0)
        chunk_ids = sorted(int(cid) for cid in ep["chunks"].keys())
        n_chunks = len(chunk_ids)
        for cpos, cid in enumerate(chunk_ids):
            ch = ep["chunks"][cid]
            batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in ch["batch"].items()}
            states = torch.from_numpy(np.frombuffer(base64.b64decode(ch["states"]), dtype=np.float32).copy()).reshape(
                -1, num_steps + 1, chunk_size, max_action_dim
            )
            vmask = torch.from_numpy(np.frombuffer(base64.b64decode(ch["valid_action_mask"]), dtype=np.bool_).copy()).reshape(
                -1, chunk_size, max_action_dim
            )
            vpos = torch.from_numpy(np.frombuffer(base64.b64decode(ch["valid_positions"]), dtype=np.bool_).copy()).reshape(-1, chunk_size)
            elp = torch.from_numpy(np.frombuffer(base64.b64decode(ch["element_log_probs"]), dtype=np.float32).copy()).reshape(
                -1, num_steps, chunk_size, max_action_dim
            )
            # collection-time per-step log-prob (the invariant rescoring must
            # reproduce: Pass A recomputes it under the round-start policy)
            stored_old = (
                elp.float() * vmask.unsqueeze(1).float() * vpos[:, None, :, None].float()
            ).sum(dim=(-1, -2))
            items.append({
                "batch": batch,
                "states": states.cuda(),
                "vmask": vmask.cuda(),
                "vpos": vpos.cuda(),
                "stored_old": stored_old.cuda(),
                "adv": float(adv),
                "eta": float(ch["eta"]),
                "weight": float(chunk_discount) ** (n_chunks - 1 - cpos) / (n_episodes * n_chunks),
            })
    nchunks_total = len(items)
    print(f"[train] decoded {nchunks_total} chunks from {n_episodes} mixed-group episodes, "
          f"steps={steps} lr={lr} ref=base chunk_discount={chunk_discount} "
          f"(batch_size forced to 1: batched rescoring corrupts logp, see docs)", flush=True)
    if nchunks_total == 0:
        raise RuntimeError("no recorded chunks to train on")

    def make_minibatches():
        # CRITICAL: one chunk per minibatch. Batching chunks with different
        # prefix caches / padded language changes the batched attention and the
        # rescored logp drifts from the collection-time per-chunk logp (measured
        # ratio 0.5-1.65 for 4-chunk batches). Both Pass A and Pass B would be
        # wrong in the SAME way so the ratio guard cannot catch it, silently
        # corrupting the objective. Rescoring per chunk matches collection
        # bit-for-bit (verified ratio == 1.0 exactly).
        for i in range(nchunks_total):
            mb = items[i:i + 1]
            batch = _stack_batches([x["batch"] for x in mb])
            states = torch.cat([x["states"] for x in mb], dim=0)
            vmask = torch.cat([x["vmask"] for x in mb], dim=0)
            vpos = torch.cat([x["vpos"] for x in mb], dim=0)
            weights = torch.tensor([x["weight"] for x in mb], dtype=torch.float32, device=states.device)
            traj = sde_sampling.SmolVLATrajectory(
                states=states, element_log_probs=states.new_zeros((0,)), valid_action_mask=vmask, eta=mb[0]["eta"]
            )
            yield mb, batch, traj, vpos, weights

    # Pass A: OLD (round-start policy) + REF (fixed base) log-probs, no grad, batched
    old_logps = []
    ref_logps = []
    with torch.no_grad(), sde_sampling.rollout_autocast("cuda"):
        for mb, batch, traj, vpos, weights in make_minibatches():
            pm, pc = sde_sampling.prepare_policy_prefix(policy, batch)
            old_lp = sde_sampling.recompute_log_probs(policy.model, pm, pc, traj, valid_positions=vpos)
            drift = (old_lp.detach() - mb[0]["stored_old"]).abs().max().item()
            if drift > RATIO_TOLERANCE:
                raise RuntimeError(
                    f"rescored old logp drifts {drift:.4f} nats from the collection-time "
                    "log-prob of the same chunk; collection and rescoring are on different "
                    "numeric paths (both sides must use rollout_autocast and identical "
                    "input dtypes, and rescoring must be per chunk)."
                )
            pmr, pcr = sde_sampling.prepare_policy_prefix(ref, batch)
            ref_lp = sde_sampling.recompute_log_probs(ref.model, pmr, pcr, traj, valid_positions=vpos)
            old_logps.append(old_lp.detach())
            ref_logps.append(ref_lp.detach())

    # Passes B: gradient steps
    checked_ratio = False
    for step_idx in range(steps):
        total_loss = 0.0
        ratio_acc = 0.0
        nchunks = 0
        for (mb, batch, traj, vpos, weights), old_lp, ref_lp in zip(make_minibatches(), old_logps, ref_logps):
            B = len(mb)
            adv_t = torch.tensor([x["adv"] for x in mb], dtype=torch.float32, device="cuda")
            with sde_sampling.rollout_autocast("cuda"):
                with torch.no_grad():
                    pm, pc = sde_sampling.prepare_policy_prefix(policy, batch)
                logp = sde_sampling.recompute_log_probs(policy.model, pm, pc, traj, valid_positions=vpos)
            logp = logp.unsqueeze(1)
            old_per_step = old_lp.unsqueeze(1)
            ref_logp = ref_lp.unsqueeze(1)
            loss, metrics = grpo.grpo_loss(logp, old_per_step, ref_logp, adv_t,
                                           clip_epsilon=clip_epsilon, kl_beta=kl_beta,
                                           sample_weights=weights)
            if not checked_ratio:
                checked_ratio = True
                drift = abs(float(metrics["ratio_mean"].cpu()) - 1.0)
                if drift > RATIO_TOLERANCE:
                    raise RuntimeError(
                        f"first-forward ratio_mean={float(metrics['ratio_mean'].cpu()):.4f} "
                        f"drifts {drift:.4f} from 1.0; collection and rescoring are on "
                        "different numeric paths (images must stay float32, both sides must "
                        "use rollout_autocast)."
                    )
            loss.backward()
            total_loss += float(loss.detach().cpu()) * B
            ratio_acc += float(metrics["ratio_mean"].cpu()) * B
            nchunks += B
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        optim.zero_grad()
        print(f"[train] step {step_idx + 1}/{steps} chunks={nchunks} loss={total_loss / max(nchunks, 1):.4f} "
              f"ratio_mean={ratio_acc / max(nchunks, 1):.4f}", flush=True)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(save_dir))
    if preprocessor is not None:
        preprocessor.save_pretrained(str(save_dir))
    if postprocessor is not None:
        postprocessor.save_pretrained(str(save_dir))
    print(f"[train] weights saved -> {save_dir} (overwrite)", flush=True)
    return {
        "status": "trained",
        "chunks": nchunks_total,
        "loss": total_loss,
        "ratio_mean": ratio_acc / max(nchunks, 1),
    }
