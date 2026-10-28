"""Resumable experiment runner.  One JSONL record per
(question, policy, schedule, repetition), written under runs/<model>/.

  python scripts/run.py --model llama3-8b --stage main_hotpotqa
  python scripts/run.py --model llama3-8b --stage main_hotpotqa --policies irrelevant
  python scripts/run.py --model llama3-8b --stage main_hotpotqa --mode single
  python scripts/run.py --model llama3-8b --stage repeat_hotpotqa --policies full --rep 1 --no-reuse
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schedsens import data, prompts                       # noqa: E402
from schedsens.evaluate import score                      # noqa: E402
from schedsens.llm import Executor, HFModel, ResponseCache  # noqa: E402
from schedsens.reports import POLICIES, CONTROL_POLICIES  # noqa: E402
from schedsens.scheduler import (SCHEDULES, Budget, no_peer_reports,  # noqa: E402
                                 run_condition, run_single)


def out_name(stage: str, mode: str, policies: list[str], rep: int) -> str:
    if mode == "single":
        return f"{stage}_single_rep{rep}.jsonl"
    if policies == ["irrelevant"]:
        return f"{stage}_irrelevant_rep{rep}.jsonl"
    return f"{stage}_rep{rep}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--mode", choices=["workflow", "single"], default="workflow")
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--config", default="configs/frozen.json")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES),
                    choices=list(POLICIES + CONTROL_POLICIES))
    ap.add_argument("--schedules", nargs="+", default=list(SCHEDULES))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--no-reuse", action="store_true", help="disable response reuse (repeat controls)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.config).read_text())
    budget = Budget(cfg["max_input"], cfg["analyst_out"], cfg["final_out"], cfg["single_max_input"])
    man = json.loads((ROOT / args.manifest).read_text())["stages"][args.stage]
    all_ids = man["ids"]
    ids = all_ids[: args.limit] if args.limit else all_ids
    exs = {e.qid: e for e in data.load(man["dataset"], ROOT / man["file"]) if e.qid in set(all_ids)}

    run_dir = ROOT / "runs" / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / out_name(args.stage, args.mode, args.policies, args.rep)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["qid"], r["policy"], r["schedule"], r["rep"]))

    model = HFModel(cfg["models"][args.model], max(budget.max_input, budget.single_max_input))
    cache = None if args.no_reuse else ResponseCache(run_dir / f"cache_{man['dataset']}.jsonl")
    execute = Executor(model, cache, reuse=not args.no_reuse)

    meta = {"stage": args.stage, "mode": args.mode, "model_key": args.model,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv,
            "config": cfg, "budget": budget.__dict__, "reuse": not args.no_reuse,
            "python": platform.python_version(), "platform": platform.platform(),
            "prompt_template_sha256": hashlib.sha256(
                (prompts.ANALYST_SYSTEM + prompts.ORCH_SYSTEM + prompts.SINGLE_SYSTEM).encode()).hexdigest(),
            "manifest_file_sha256": man["sha256"], **model.info()}
    with out.with_suffix(".meta.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

    if args.mode == "single":
        todo = [(q, "single_agent", "single") for q in ids if (q, "single_agent", "single", args.rep) not in done]
    else:
        todo = [(q, p, s) for q in ids for p in args.policies for s in args.schedules
                if (q, p, s, args.rep) not in done]
    print(f"{len(done)} done, {len(todo)} to run -> {out}", flush=True)

    donors: dict[str, dict] = {}
    t_start, executed = time.perf_counter(), 0
    with out.open("a", encoding="utf-8") as f:
        for i, (qid, pol, sch) in enumerate(todo, 1):
            ex = exs[qid]
            t0 = time.perf_counter()
            if args.mode == "single":
                rec = run_single(ex, args.rep, execute, model.template, budget)
            else:
                donor = None
                if pol == "irrelevant":
                    # donor: next question in the frozen manifest order (cyclic)
                    dq = all_ids[(all_ids.index(qid) + 1) % len(all_ids)]
                    if dq not in donors:
                        donors[dq] = no_peer_reports(exs[dq], execute, model.template, budget)
                    donor = donors[dq]
                rec = run_condition(ex, pol, sch, args.rep, execute, model.template, model.count,
                                    budget, donor)
                if donor is not None:
                    rec["donor_qid"] = dq
                rec["analyst_scores"] = {a: score(ex, rec["analyst_answers"][a], [])["f1"] for a in "AB"}
            rec["model"] = args.model
            rec["wall_seconds"] = time.perf_counter() - t0
            rec["score"] = score(ex, rec["final_answer"], rec["final_support"])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            executed += sum(not c["cached"] for c in rec["calls"])
            el = time.perf_counter() - t_start
            print(f"[{i}/{len(todo)}] {args.model} {args.stage} {qid[:12]} {pol:13s} {sch:13s} "
                  f"f1={rec['score']['f1']:.2f} exec={executed} elapsed={el/60:.1f}m "
                  f"eta={el/i*(len(todo)-i)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
