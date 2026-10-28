"""Re-execute logged conditions without response reuse and compare prompts
and outputs byte for byte with the original log."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schedsens import data                                 # noqa: E402
from schedsens.llm import Executor, HFModel                 # noqa: E402
from schedsens.scheduler import Budget, run_condition       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3-8b")
    ap.add_argument("--log", default="runs/llama3-8b/main_hotpotqa_rep0.jsonl")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    cfg = json.loads((ROOT / "configs/frozen.json").read_text())
    budget = Budget(cfg["max_input"], cfg["analyst_out"], cfg["final_out"], cfg["single_max_input"])
    log = ROOT / args.log
    f = log.open(encoding="utf-8") if log.exists() else gzip.open(f"{log}.gz", "rt", encoding="utf-8")
    with f:
        recs = [json.loads(l) for l in f if l.strip()]
    picks = [r for r in recs if r["policy"] == "full" and r["schedule"] == "a_first"][: args.n]
    ds = picks[0]["dataset"]
    src = {"hotpotqa": "data/raw/hotpot_distractor_validation.parquet",
           "musique": "data/raw/musique_ans_v1.0_dev.jsonl",
           "2wiki": "data/raw/2wiki/dev.json"}[ds]
    want = {r["qid"] for r in picks}
    exs = {e.qid: e for e in data.load(ds, ROOT / src) if e.qid in want}
    model = HFModel(cfg["models"][args.model], max(budget.max_input, budget.single_max_input))
    execute = Executor(model, None, reuse=False)
    same_p = same_o = total = 0
    for r in picks:
        new = run_condition(exs[r["qid"]], r["policy"], r["schedule"], 0, execute,
                            model.template, model.count, budget)
        for a, b in zip(r["calls"], new["calls"]):
            total += 1
            same_p += a["prompt_hash"] == b["prompt_hash"]
            same_o += a["output"] == b["output"]
    print(f"prompts identical {same_p}/{total}, outputs identical {same_o}/{total}")


if __name__ == "__main__":
    main()
