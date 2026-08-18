#!/usr/bin/env python
"""Parse lerobot_eval results: single eval_info.json or a directory of task_*/eval_info.json."""
import json
import sys
from pathlib import Path


def load_info(p: Path) -> dict:
    return json.loads(p.read_text())


def summarize(info: dict) -> tuple[int, float, list[dict]]:
    per_task = []
    for t in info.get("per_task", []):
        tid = t.get("task_id")
        group = t.get("task_group", "?")
        suc = t.get("metrics", {}).get("successes", [])
        per_task.append({"task_id": tid, "group": group, "successes": suc})
    overall = info.get("overall", {})
    return overall.get("n_episodes", 0), overall.get("pc_success", float("nan")), per_task


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: parse_eval_results.py <eval_info.json | dir>")
        sys.exit(1)
    p = Path(sys.argv[1])
    if p.is_dir():
        files = sorted(p.rglob("eval_info.json"))
        if not files:
            print(f"ERROR: no eval_info.json under {p}")
            sys.exit(1)
        all_per_task = []
        total_ep = 0
        total_ok = 0
        for f in files:
            n, pc, pts = summarize(load_info(f))
            total_ep += n
            for pt in pts:
                all_per_task.append(pt)
                total_ok += sum(1 for s in pt["successes"] if s)
        overall_pc = 100.0 * total_ok / total_ep if total_ep else float("nan")
        per_task = all_per_task
        eval_s = float("nan")
        ep_s = float("nan")
    else:
        n, overall_pc, per_task = summarize(load_info(p))
        total_ep = n
        eval_s = load_info(p).get("overall", {}).get("eval_s", 0)
        ep_s = load_info(p).get("overall", {}).get("eval_ep_s", 0)

    print("=" * 70)
    print("LIBERO spatial evaluation summary (per-task)")
    print("=" * 70)
    print(f"{'task_id':>8} {'group':<20} {'success':>18} {'pc':>8}")
    print("-" * 70)
    for t in sorted(per_task, key=lambda x: (x.get("task_id") is None, x.get("task_id") or -1)):
        tid = t.get("task_id")
        group = t.get("group", "?")
        suc = t.get("successes", [])
        n_ok = sum(1 for s in suc if s)
        n = len(suc)
        pc = (100.0 * n_ok / n) if n else float("nan")
        print(f"{str(tid):>8} {group:<20} {n_ok:>5}/{n:<12} {pc:>7.1f}%")
    print("-" * 70)
    print(f"OVERALL  {total_ep} episodes  pc_success={overall_pc:.1f}%  " +
          (f"time={eval_s/60:.1f}min ({ep_s:.1f}s/ep)" if eval_s == eval_s else ""))
    print("SUMMARY_JSON " + json.dumps({
        "n_episodes": total_ep, "pc_success": overall_pc, "per_task": per_task}))
    print("=" * 70)


if __name__ == "__main__":
    main()