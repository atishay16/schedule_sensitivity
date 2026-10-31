"""results/summary.json -> LaTeX tables in paper/ (every number in the
tables comes from the summary; nothing is typed by hand)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("llama3-8b", "Llama-3-8B"), ("qwen2.5-7b", "Qwen2.5-7B")]
DATASETS = [("hotpotqa", "HotpotQA"), ("musique", "MuSiQue"), ("2wiki", "2Wiki")]
SHORT_M = {"llama3-8b": "Llama", "qwen2.5-7b": "Qwen"}
SHORT_D = {"hotpotqa": "Hotpot", "musique": "MuSiQue", "2wiki": "2Wiki"}
POL = [("full", "Full"), ("evidence_only", "Evid.-only"), ("isolation", "Isolation")]


def p(x, d=1):
    s = f"{x*100:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def ci(c):
    return f"{p(c[0])} {{\\tiny[{p(c[1])}, {p(c[2])}]}}"


def pm(c):
    """Signed point estimate with CI, for differences."""
    s = p(c[0])
    s = s if s.startswith("$-$") else "+" + s
    return f"{s} {{\\tiny[{p(c[1])}, {p(c[2])}]}}"


def wrap(body, caption, label, cols, wide=False):
    env = "table*" if wide else "table"
    width = "\\textwidth" if wide else "\\columnwidth"
    return "\n".join([f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}", "\\setlength{\\tabcolsep}{3pt}", "\\footnotesize",
                      # wide tables fit at natural size; narrow ones shrink to the column
                      *([] if wide else [f"\\resizebox{{{width}}}{{!}}{{%"]),
                      f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
                      *body, "\\bottomrule", "\\end{tabular}" + ("" if wide else "}"), f"\\end{{{env}}}"])


def cells(S):
    for mk, mn in MODELS:
        for dk, dn in DATASETS:
            if mk in S and dk in S[mk]:
                yield mk, mn, dk, dn, S[mk][dk]


def main_table(S):
    body = [r"Model & Data ($n$) & Policy & EM & F1 & SP-F1 & Worst F1 & Best F1 & Dis.\ (\%) & Inst.\ (\%) \\", r"\midrule"]
    rows = list(cells(S))
    for i, (mk, mn, dk, dn, R) in enumerate(rows):
        for j, (pk, pn) in enumerate(POL):
            d = R["policies"][pk]
            first_m = f"\\multirow{{9}}{{*}}{{{mn}}}" if j == 0 and (i % 3 == 0) else ""
            first_d = f"\\multirow{{3}}{{*}}{{{dn} ({R['n_questions']})}}" if j == 0 else ""
            body.append(f"{first_m} & {first_d} & {pn} & {p(d['em']['sched_avg'][0])} & {ci(d['f1']['sched_avg'])} & "
                        f"{p(d['sp_f1']['sched_avg'][0])} & {p(d['f1']['worst_schedule'][0])} & "
                        f"{p(d['f1']['best_schedule'])} & {ci(d['disagreement'])} & {p(d['em_instability'][0])} \\\\")
        if i < len(rows) - 1:
            body.append(r"\cmidrule(lr){2-10}" if i % 3 != 2 else r"\midrule")
    return wrap(body,
                r"Answer quality and schedule sensitivity. Scores are averaged over the three run orders of each question; brackets are 95\% bootstrap confidence intervals. Worst/Best: lowest/highest F1 over the three orders, averaged over questions. Dis.: questions whose final answer is not the same in all three orders. Inst.: questions answered correctly (EM) in one order and incorrectly in another.",
                "tab:main", "lllccccccc", wide=True)


def paired_table(S):
    body = [r" & & \multicolumn{2}{c}{$\Delta$F1} & $\Delta$Worst F1 & $\Delta$Dis. \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-5}\cmidrule(lr){6-6}",
            r"Model & Data & Full$-$Iso. & Evid.$-$Iso. & Full$-$Iso. & Full$-$Evid. \\", r"\midrule"]
    rows = list(cells(S))
    for i, (mk, mn, dk, dn, R) in enumerate(rows):
        P = R["paired"]
        first = f"\\multirow{{3}}{{*}}{{{SHORT_M[mk]}}}" if i % 3 == 0 else ""
        body.append(f"{first} & {SHORT_D[dk]} &{pm(P['full-isolation']['f1'])} & {pm(P['evidence_only-isolation']['f1'])} & "
                    f"{pm(P['full-isolation']['worst_f1'])} & {pm(P['full-evidence_only']['disagreement'])} \\\\")
        if i == 2:
            body.append(r"\midrule")
    return wrap(body, r"Differences between settings, computed question by question (points; 95\% bootstrap confidence intervals). F1 is averaged over the three run orders first; worst F1 is each question's lowest F1 over the orders. The last column compares disagreement under full and evidence-only sharing.",
                "tab:paired", "llcccc")


def control_table(S):
    body = [r"Model & Data & \multicolumn{3}{c}{Disagreement (\%)} & \multicolumn{2}{c}{Changed (\%)} & \multicolumn{3}{c}{Adopted (\%)} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}",
            r" & & Full & Irr. & Full$-$Irr. & Full & Irr. & Full & Evid. & Irr. \\", r"\midrule"]
    rows = list(cells(S))
    for i, (mk, mn, dk, dn, R) in enumerate(rows):
        if "irrelevant" not in R:
            continue
        A = R["analyst_influence"]
        first = f"\\multirow{{3}}{{*}}{{{mn}}}" if i % 3 == 0 else ""
        body.append(f"{first} & {dn} & {p(R['policies']['full']['disagreement'][0])} & "
                    f"{p(R['irrelevant']['disagreement'][0])} & {pm(R['irrelevant']['full_minus_irrelevant_disagreement'])} & "
                    f"{p(A['full']['changed'][0])} & {p(A['irrelevant']['changed'][0])} & "
                    f"{p(A['full']['adopted'][0])} & {p(A['evidence_only']['adopted'][0])} & {p(A['irrelevant']['adopted'][0])} \\\\")
        if i == 2:
            body.append(r"\midrule")
    return wrap(body, r"Peer content versus peer presence. Irr.: the later analyst is shown a real report written for a different question, with the same timing as full sharing. Changed: the later analyst's tentative answer differs from the one it gives without a peer. Adopted: it switches to the answer in the report it was shown. Rates cover A-first and B-first; brackets are 95\% bootstrap confidence intervals.",
                "tab:control", "llcccccccc", wide=True)


def baseline_table(S):
    body = [r" & & \multicolumn{3}{c}{Answer F1} & \multicolumn{2}{c}{Repeat subset (\%)} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
            r"Model & Data & Single & Iso. & Iso.$-$Single & Repeat diff. & Across orders \\", r"\midrule"]
    rows = list(cells(S))
    for i, (mk, mn, dk, dn, R) in enumerate(rows):
        if "single_agent" not in R:
            continue
        sa = R["single_agent"]
        rp = R.get("repeat")
        rep_cells = (f"{p(rp['raw_output_diff_rate'])} ({rp['n_calls']}) & {ci(rp['cross_schedule_disagreement_same_qs'])}"
                     if rp else "-- & --")
        first = f"\\multirow{{3}}{{*}}{{{SHORT_M[mk]}}}" if i % 3 == 0 else ""
        body.append(f"{first} & {SHORT_D[dk]} & {p(sa['f1'][0])} & {p(R['policies']['isolation']['f1']['sched_avg'][0])} & "
                    f"{pm(sa['isolation_minus_single_f1'])} & {rep_cells} \\\\")
        if i == 2:
            body.append(r"\midrule")
    return wrap(body, r"Reference points. Single: one model call that reads all the evidence. Iso.: the two-analyst workflow with isolation. Repeat diff.: share of calls whose output changed when the same run was repeated (number of calls in parentheses), compared with disagreement across run orders on the same questions.",
                "tab:baseline", "llccccc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--outdir", default="paper")
    args = ap.parse_args()
    S = json.loads((ROOT / args.summary).read_text())
    od = ROOT / args.outdir
    od.mkdir(parents=True, exist_ok=True)
    for name, fn in (("tab_main", main_table), ("tab_paired", paired_table),
                     ("tab_control", control_table), ("tab_baseline", baseline_table)):
        (od / f"{name}.tex").write_text(fn(S), encoding="utf-8")
    print("tables written to", od)


if __name__ == "__main__":
    main()
