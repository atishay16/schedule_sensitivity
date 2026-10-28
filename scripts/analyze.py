"""Paired, question-level analysis of run logs -> results/summary.json.

Schedules are never treated as independent examples: scores are averaged over
schedules within a question before policy comparisons, and the bootstrap
resamples questions with all of their conditions together."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from schedsens.evaluate import normalize_answer   # noqa: E402

POLICIES = ("full", "evidence_only", "isolation")
GRID_POLICIES = POLICIES + ("irrelevant",)
SCHEDS = ("a_first", "b_first", "both_captured")
PNAME = {"full": "Full eager", "evidence_only": "Evidence-only", "isolation": "Isolation",
         "irrelevant": "Irrelevant peer"}
SNAME = {"a_first": "A-first", "b_first": "B-first", "both_captured": "Both-captured"}
MODELS = ("llama3-8b", "qwen2.5-7b")
DATASETS = ("hotpotqa", "musique", "2wiki")
B = 10000


def load(path: Path) -> list[dict]:
    """Read a JSONL run log, falling back to the gzip copy kept in the repository."""
    if path.exists():
        f = path.open(encoding="utf-8")
    elif (gz := path.with_name(path.name + ".gz")).exists():
        f = gzip.open(gz, "rt", encoding="utf-8")
    else:
        return []
    with f:
        return [json.loads(l) for l in f if l.strip()]


def norm(a):
    return normalize_answer(a) if a is not None else "<missing>"


def boot_ci(per_q: np.ndarray, rng, stat=np.mean):
    n = len(per_q)
    idx = rng.integers(0, n, size=(B, n))
    s = stat(per_q[idx], axis=1)
    return [float(stat(per_q)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


class Grid:
    """question x policy x schedule view of workflow records."""

    def __init__(self, recs: list[dict], qids: list[str], policies):
        self.qids, self.policies = qids, policies
        self.by = {(r["qid"], r["policy"], r["schedule"]): r for r in recs}
        missing = [(q, p, s) for q in qids for p in policies for s in SCHEDS if (q, p, s) not in self.by]
        assert not missing, f"incomplete grid, e.g. {missing[:3]} ({len(missing)} missing)"

    def arr(self, metric) -> np.ndarray:
        return np.array([[[metric(self.by[(q, p, s)]) for s in SCHEDS] for p in self.policies]
                         for q in self.qids], float)

    def disagree(self, p) -> np.ndarray:
        return np.array([len({norm(self.by[(q, p, s)]["final_answer"]) for s in SCHEDS}) > 1
                         for q in self.qids], float)


def analyst_answers(rec, role):
    return norm(rec["analyst_answers"][role])


def influence(grid: Grid, policy: str, solo: dict, rng) -> dict:
    """Later-starting analyst under peer-visible schedules, relative to its own
    no-peer answer: answer changed, and switched to the visible report's answer."""
    ch, ad = [], []
    for q in grid.qids:
        c_q, a_q = [], []
        for s in ("a_first", "b_first"):
            r = grid.by[(q, policy, s)]
            later = "B" if s == "a_first" else "A"
            first = "A" if later == "B" else "B"
            la, sl = analyst_answers(r, later), solo[(q, later)]
            seen = norm(next(c for c in r["calls"] if c["role"] == f"analyst_{first}")["parsed"]["answer"])
            if policy == "irrelevant":
                seen = r["_donor_answer"][first]
            c_q.append(la != sl)
            a_q.append(la == seen and sl != seen)
        ch.append(np.mean(c_q))
        ad.append(np.mean(a_q))
    return {"changed": np.array(ch), "adopted": np.array(ad)}


def analyze_cell(main, irr, single, rep, qids, rng) -> dict:
    out = {"n_questions": len(qids)}
    g = Grid(main, qids, POLICIES)
    hot = main[0]["dataset"] in ("hotpotqa", "2wiki")
    metrics = {"em": lambda r: r["score"]["em"], "f1": lambda r: r["score"]["f1"],
               "sp_f1": lambda r: r["score"]["sp_f1"]}
    if hot:
        metrics["joint_f1"] = lambda r: r["score"]["joint_f1"]
    G = {k: g.arr(f) for k, f in metrics.items()}

    pol = {}
    for pi, p in enumerate(POLICIES):
        d = {k: {"per_schedule": {s: float(a[:, pi, si].mean()) for si, s in enumerate(SCHEDS)},
                 "sched_avg": boot_ci(a[:, pi, :].mean(1), rng),
                 "worst_schedule": boot_ci(a[:, pi, :].min(1), rng),
                 "best_schedule": float(a[:, pi, :].max(1).mean())} for k, a in G.items()}
        d["disagreement"] = boot_ci(g.disagree(p), rng)
        em = G["em"][:, pi, :]
        d["em_instability"] = boot_ci(((em.max(1) == 1) & (em.min(1) == 0)).astype(float), rng)
        pol[p] = d
    out["policies"] = pol

    pairs = {}
    for a, b in (("full", "isolation"), ("evidence_only", "isolation"), ("full", "evidence_only")):
        ia, ib = POLICIES.index(a), POLICIES.index(b)
        pairs[f"{a}-{b}"] = {
            "f1": boot_ci(G["f1"][:, ia, :].mean(1) - G["f1"][:, ib, :].mean(1), rng),
            "em": boot_ci(G["em"][:, ia, :].mean(1) - G["em"][:, ib, :].mean(1), rng),
            "worst_f1": boot_ci(G["f1"][:, ia, :].min(1) - G["f1"][:, ib, :].min(1), rng),
            "disagreement": boot_ci(g.disagree(a) - g.disagree(b), rng)}
    out["paired"] = pairs

    flips = {}
    ref = SCHEDS.index("both_captured")
    for p in ("full", "evidence_only"):
        pi = POLICIES.index(p)
        for s in ("a_first", "b_first"):
            si = SCHEDS.index(s)
            r_em, c_em = G["em"][:, pi, ref], G["em"][:, pi, si]
            flips[f"{p}/{s}"] = {"gain": int(((r_em == 0) & (c_em == 1)).sum()),
                                 "loss": int(((r_em == 1) & (c_em == 0)).sum()),
                                 "f1_delta": boot_ci(G["f1"][:, pi, si] - G["f1"][:, pi, ref], rng)}
    out["flips_vs_no_peer"] = flips

    solo = {}
    for q in qids:
        r = g.by[(q, "isolation", "both_captured")]
        for role in "AB":
            solo[(q, role)] = analyst_answers(r, role)
    infl = {p: influence(g, p, solo, rng) for p in ("full", "evidence_only")}

    # irrelevant-peer control: same timing as full sharing, unrelated report content
    if irr:
        gi = Grid(irr, qids, ("irrelevant",))
        for r in irr:
            r["_donor_answer"] = {}
        donors = {r["qid"]: r for r in main if r["policy"] == "isolation" and r["schedule"] == "both_captured"}
        for r in gi.by.values():
            d = donors.get(r.get("donor_qid"))
            r["_donor_answer"] = {role: analyst_answers(d, role) for role in "AB"} if d else {"A": "", "B": ""}
        infl["irrelevant"] = influence(gi, "irrelevant", solo, rng)
        f1_i = gi.arr(lambda r: r["score"]["f1"])[:, 0, :]
        em_i = gi.arr(lambda r: r["score"]["em"])[:, 0, :]
        dis_i = gi.disagree("irrelevant")
        out["irrelevant"] = {
            "f1_sched_avg": boot_ci(f1_i.mean(1), rng),
            "per_schedule": {s: float(f1_i[:, si].mean()) for si, s in enumerate(SCHEDS)},
            "disagreement": boot_ci(dis_i, rng),
            "em_instability": boot_ci(((em_i.max(1) == 1) & (em_i.min(1) == 0)).astype(float), rng),
            "full_minus_irrelevant_disagreement": boot_ci(g.disagree("full") - dis_i, rng),
            "full_minus_irrelevant_f1": boot_ci(G["f1"][:, 0, :].mean(1) - f1_i.mean(1), rng),
            "irrelevant_minus_isolation_f1": boot_ci(f1_i.mean(1) - G["f1"][:, 2, :].mean(1), rng),
        }
    out["analyst_influence"] = {
        p: {"changed": boot_ci(v["changed"], rng), "adopted": boot_ci(v["adopted"], rng)}
        for p, v in infl.items()}
    out["analyst_influence"]["adoption_full_minus_evidence_only"] = boot_ci(
        infl["full"]["adopted"] - infl["evidence_only"]["adopted"], rng)
    if "irrelevant" in infl:
        out["analyst_influence"]["changed_full_minus_irrelevant"] = boot_ci(
            infl["full"]["changed"] - infl["irrelevant"]["changed"], rng)

    # single-agent reference over all evidence
    if single:
        sb = {r["qid"]: r for r in single}
        assert all(q in sb for q in qids), "incomplete single-agent run"
        sf1 = np.array([sb[q]["score"]["f1"] for q in qids])
        sem = np.array([sb[q]["score"]["em"] for q in qids])
        ssp = np.array([sb[q]["score"]["sp_f1"] for q in qids])
        iso = POLICIES.index("isolation")
        out["single_agent"] = {
            "em": boot_ci(sem, rng), "f1": boot_ci(sf1, rng), "sp_f1": boot_ci(ssp, rng),
            "isolation_minus_single_f1": boot_ci(G["f1"][:, iso, :].mean(1) - sf1, rng),
            "full_minus_single_f1": boot_ci(G["f1"][:, 0, :].mean(1) - sf1, rng),
            "missing": sum(r["final_answer"] is None for r in single),
            "in_tokens": float(np.mean([r["calls"][0]["in_tokens"] for r in single]))}

    # identical-condition repeat (reuse disabled) vs the main run
    if rep:
        rq = sorted({r["qid"] for r in rep})
        diff_ans, diff_raw = [], []
        for r in rep:
            m = g.by[(r["qid"], "full", r["schedule"])]
            diff_ans.append(norm(r["final_answer"]) != norm(m["final_answer"]))
            diff_raw += [c["output"] != mc["output"] for c, mc in zip(r["calls"], m["calls"])]
        cross = np.array([len({norm(g.by[(q, "full", s)]["final_answer"]) for s in SCHEDS}) > 1 for q in rq], float)
        out["repeat"] = {"n_questions": len(rq), "n_calls": len(diff_raw),
                         "answer_diff_rate": float(np.mean(diff_ans)),
                         "raw_output_diff_rate": float(np.mean(diff_raw)),
                         "cross_schedule_disagreement_same_qs": boot_ci(cross, rng)}

    cost = {}
    for p in POLICIES:
        rs = [g.by[(q, p, s)] for q in qids for s in SCHEDS]
        calls = [c for r in rs for c in r["calls"]]
        an = [c for c in calls if c["role"].startswith("analyst")]
        cost[p] = {"in_tokens_per_run": sum(c["in_tokens"] for c in calls) / len(rs),
                   "out_tokens_per_run": sum(c["out_tokens"] for c in calls) / len(rs),
                   "gen_seconds_per_run": sum(c["seconds"] for c in calls) / len(rs),
                   "unique_calls_per_q": len({c["prompt_hash"] for c in calls}) / len(qids),
                   "invalid_reports": sum(not c["valid"] for c in an),
                   "analyst_calls": len(an),
                   "findings_missing_ids": sum(c["findings_missing_ids"] for c in an),
                   "missing_final": sum(r["final_answer"] is None for r in rs),
                   "over_budget": sum(c["over_budget"] for c in calls)}
    out["cost"] = cost
    return out


def fmt(ci):
    return f"{ci[0]*100:.1f} [{ci[1]*100:.1f}, {ci[2]*100:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--stage-prefix", default="main")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    man = json.loads((ROOT / "data" / "manifest.json").read_text())["stages"]
    rng = np.random.default_rng(12345)
    res = {}
    for m in args.models:
        rd = ROOT / "runs" / m
        for ds in args.datasets:
            st = f"{args.stage_prefix}_{ds}"
            qids = man[st]["ids"]
            keep = lambda rs: [r for r in rs if r["qid"] in set(qids)]
            main_r = keep(load(rd / f"{st}_rep0.jsonl"))
            if not main_r:
                continue
            res.setdefault(m, {})[ds] = analyze_cell(
                main_r, keep(load(rd / f"{st}_irrelevant_rep0.jsonl")),
                keep(load(rd / f"{st}_single_rep0.jsonl")),
                load(rd / f"repeat_{ds}_rep1.jsonl") if args.stage_prefix == "main" else [],
                qids, rng)
            R = res[m][ds]
            print(f"\n== {m} / {ds} (n={R['n_questions']})")
            for p, d in R["policies"].items():
                print(f"  {PNAME[p]:14s} F1 {fmt(d['f1']['sched_avg'])}  dis {fmt(d['disagreement'])}  "
                      f"inst {fmt(d['em_instability'])}  worstF1 {fmt(d['f1']['worst_schedule'])}")
            for k, e in R["paired"].items():
                print(f"  {k:24s} dF1 {fmt(e['f1'])}  dDis {fmt(e['disagreement'])}")
            if "irrelevant" in R:
                print(f"  irrelevant: dis {fmt(R['irrelevant']['disagreement'])}  "
                      f"full-irr dis {fmt(R['irrelevant']['full_minus_irrelevant_disagreement'])}")
            print("  influence", {k: (fmt(v['changed']), fmt(v['adopted'])) if isinstance(v, dict) else fmt(v)
                                  for k, v in R["analyst_influence"].items()})
            if "single_agent" in R:
                print(f"  single F1 {fmt(R['single_agent']['f1'])}  iso-single {fmt(R['single_agent']['isolation_minus_single_f1'])}")
            if "repeat" in R:
                print(f"  repeat: raw diff {R['repeat']['raw_output_diff_rate']:.3f} over {R['repeat']['n_calls']} calls; "
                      f"cross-schedule {fmt(R['repeat']['cross_schedule_disagreement_same_qs'])}")
    od = ROOT / args.out
    od.mkdir(parents=True, exist_ok=True)
    (od / "summary.json").write_text(json.dumps(res, indent=1))
    print("\nwrote", od / "summary.json")


if __name__ == "__main__":
    main()
