"""Freeze seeded question manifests before any inference.

Eligibility uses only structural validity and a predeclared token-budget
check (own evidence + a reserved worst-case peer report must fit).  It never
looks at accuracy, schedule effects, or gold evidence placement."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schedsens import data, prompts                      # noqa: E402
from schedsens.reports import NO_PEER                    # noqa: E402

STAGES = {
    # name: (dataset, file, n, max records to scan or None)
    "pilot_hotpotqa": ("hotpotqa", "data/raw/hotpot_distractor_train_0.parquet", 10, None),
    "pilot_musique": ("musique", "data/raw/musique_ans_v1.0_train.jsonl", 5, None),
    "pilot_2wiki": ("2wiki", "data/raw/2wiki/train.json", 10, 5000),
    "main_hotpotqa": ("hotpotqa", "data/raw/hotpot_distractor_validation.parquet", 500, None),
    "main_musique": ("musique", "data/raw/musique_ans_v1.0_dev.jsonl", 300, None),
    "main_2wiki": ("2wiki", "data/raw/2wiki/dev.json", 300, None),
}
REPEAT = {"hotpotqa": 50, "musique": 30, "2wiki": 30}   # predetermined: first N of each main order
SEED = 20270101
TOKENIZERS = ("unsloth/llama-3-8b-Instruct-bnb-4bit", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-input", type=int, default=3500)
    ap.add_argument("--single-max-input", type=int, default=6000)
    ap.add_argument("--peer-reserve", type=int, default=320,
                    help="tokens reserved for the largest allowed rendered peer report")
    ap.add_argument("--out", default="data/manifest.json")
    args = ap.parse_args()

    from itertools import islice
    from transformers import AutoTokenizer
    toks = [AutoTokenizer.from_pretrained(m) for m in TOKENIZERS]

    def worst_tokens(ex):
        """Max over tokenizers of (analyst prompt + reserved peer report) and
        of the single-agent prompt over all evidence."""
        analyst = single = 0
        for tok in toks:
            count = lambda msgs: len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
            no_peer_len = len(tok(NO_PEER, add_special_tokens=False)["input_ids"])
            for agent, paras in zip("AB", data.shard(ex)):
                t = count(prompts.analyst_messages(ex, agent, paras, NO_PEER))
                analyst = max(analyst, t - no_peer_len + args.peer_reserve)
            single = max(single, count(prompts.single_messages(ex)))
        return analyst, single

    manifest = {"seed": SEED, "max_input": args.max_input, "single_max_input": args.single_max_input,
                "peer_reserve": args.peer_reserve, "tokenizers": list(TOKENIZERS), "stages": {}}
    for name, (ds, rel, n, scan) in STAGES.items():
        path = ROOT / rel
        exs = {e.qid: e for e in islice(data.load(ds, path), scan)}
        order = sorted(exs)
        random.Random(f"{SEED}:{name}").shuffle(order)
        chosen, excluded, toks_a, toks_s = [], [], {}, {}
        for qid in order:
            if len(chosen) == n:
                break
            ex = exs[qid]
            reason = data.structural_check(ex)
            if reason is None:
                a, sgl = worst_tokens(ex)
                if a > args.max_input:
                    reason = f"token_budget:{a}>{args.max_input}"
                elif sgl > args.single_max_input:
                    reason = f"single_token_budget:{sgl}>{args.single_max_input}"
                else:
                    toks_a[qid], toks_s[qid] = a, sgl
            if reason:
                excluded.append({"id": qid, "reason": reason})
            else:
                chosen.append(qid)
        assert len(chosen) == n, (name, len(chosen))
        manifest["stages"][name] = {"dataset": ds, "file": rel, "sha256": sha256(path),
                                    "scan_limit": scan, "ids": chosen, "excluded": excluded,
                                    "worst_case_analyst_tokens": toks_a, "single_tokens": toks_s}
        print(f"{name}: {len(chosen)} chosen, {len(excluded)} excluded, "
              f"max analyst {max(toks_a.values())}, max single {max(toks_s.values())}", flush=True)
    for ds, k in REPEAT.items():
        main_stage = manifest["stages"][f"main_{ds}"]
        manifest["stages"][f"repeat_{ds}"] = {
            **{x: main_stage[x] for x in ("dataset", "file", "sha256")},
            "ids": main_stage["ids"][:k], "excluded": []}
    out = ROOT / args.out
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
