"""Per-question scoring with the *official* HotpotQA, MuSiQue and
2WikiMultiHopQA metric code (vendored unchanged under third_party/).
Missing answers score as failures."""
from __future__ import annotations

import sys
from pathlib import Path

from .data import Example
from .reports import ID_RE

_TP = Path(__file__).resolve().parent.parent / "third_party"
sys.path.insert(0, str(_TP))
sys.path.insert(0, str(_TP / "musique"))
import hotpot_evaluate_v1 as hotpot_eval            # noqa: E402
from metrics.answer import AnswerMetric             # noqa: E402
from metrics.support import SupportMetric           # noqa: E402
import importlib.util                               # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "wiki2_eval", _TP / "2wiki" / "2wikimultihop_evaluate_v1.1.py")
wiki2_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wiki2_eval)

normalize_answer = hotpot_eval.normalize_answer


def _parse_id(tok: str) -> tuple[int, int | None]:
    m = ID_RE.fullmatch(tok)
    return int(m.group(1)), (int(m.group(2)) if m.group(2) is not None else None)


def support_units(ex: Example, ids: list[str]) -> tuple[list, int]:
    """Convert predicted source IDs to dataset support units.
    Returns (units, n_invalid_ids)."""
    by_pid = {p.pid: p for p in ex.paragraphs}
    units, bad = [], 0
    for tok in ids:
        pid, sid = _parse_id(tok)
        p = by_pid.get(pid)
        if p is None:
            bad += 1
            continue
        if ex.dataset in ("hotpotqa", "2wiki"):
            if sid is None or sid >= len(p.sentences):
                bad += 1
                continue
            u = [p.title, sid]
        else:
            u = pid
        if u not in units:
            units.append(u)
    return units, bad


def score(ex: Example, answer: str | None, support_ids: list[str]) -> dict:
    pred = answer if answer is not None else ""
    units, bad = support_units(ex, support_ids)
    if ex.dataset == "hotpotqa":
        m = {k: 0.0 for k in ("em", "f1", "prec", "recall", "sp_em", "sp_f1", "sp_prec", "sp_recall")}
        em, prec, rec = hotpot_eval.update_answer(m, pred, ex.answer)
        sp_em, sp_prec, sp_rec = hotpot_eval.update_sp(m, units, [list(g) for g in ex.gold_support])
        jp, jr = prec * sp_prec, rec * sp_rec
        joint_f1 = 2 * jp * jr / (jp + jr) if jp + jr > 0 else 0.0
        return {"em": float(m["em"]), "f1": float(m["f1"]), "sp_em": m["sp_em"],
                "sp_f1": m["sp_f1"], "joint_em": float(em) * sp_em, "joint_f1": joint_f1,
                "invalid_support_ids": bad, "missing_answer": answer is None}
    if ex.dataset == "2wiki":
        # official v1.1 scoring: gold span plus id_aliases (aliases + demonyms)
        m = {k: 0.0 for k in ("em", "f1", "prec", "recall", "sp_em", "sp_f1", "sp_prec", "sp_recall")}
        em, prec, rec = wiki2_eval.update_answer(m, pred, {ex.answer, *ex.answer_aliases})
        sp_em, sp_prec, sp_rec = wiki2_eval.update_sp(m, units, [list(g) for g in ex.gold_support])
        jp, jr = prec * sp_prec, rec * sp_rec
        joint_f1 = 2 * jp * jr / (jp + jr) if jp + jr > 0 else 0.0
        return {"em": float(m["em"]), "f1": float(m["f1"]), "sp_em": m["sp_em"],
                "sp_f1": m["sp_f1"], "joint_em": float(em) * sp_em, "joint_f1": joint_f1,
                "invalid_support_ids": bad, "missing_answer": answer is None}
    am, sm = AnswerMetric(), SupportMetric()
    am(pred, [ex.answer, *ex.answer_aliases])
    sm(units, list(ex.gold_support))
    a_em, a_f1 = am.get_metric()
    s_em, s_f1 = sm.get_metric()
    return {"em": a_em, "f1": a_f1, "sp_em": s_em, "sp_f1": s_f1,
            "invalid_support_ids": bad, "missing_answer": answer is None}
