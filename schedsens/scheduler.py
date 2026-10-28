"""Logical scheduler.  GPU calls run serially; the schedule controls when
each analyst's prompt is *captured* relative to its peer's publication.
A captured prompt is an immutable string: later writes cannot alter it."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import prompts
from .data import Example, shard
from .reports import ReportStore, parse_report, parse_final, peer_section, render_full

SCHEDULES = {
    "a_first": (("capture", "A"), ("infer", "A"), ("publish", "A"),
                ("capture", "B"), ("infer", "B"), ("publish", "B")),
    "b_first": (("capture", "B"), ("infer", "B"), ("publish", "B"),
                ("capture", "A"), ("infer", "A"), ("publish", "A")),
    "both_captured": (("capture", "A"), ("capture", "B"),
                      ("infer", "A"), ("infer", "B"),
                      ("publish", "A"), ("publish", "B")),
}


@dataclass(frozen=True)
class Budget:
    max_input: int = 3500
    analyst_out: int = 256
    final_out: int = 128
    single_max_input: int = 6000   # single-agent reference reads all evidence


@dataclass(frozen=True)
class Captured:
    agent: str
    prompt: str
    visible_report_ids: tuple[str, ...]
    peer_text: str
    captured_after: tuple[str, ...]       # agents already published at capture time


def seed_for(qid: str, agent: str, rep: int) -> int:
    return int(hashlib.sha256(f"{qid}|{agent}|{rep}".encode()).hexdigest()[:8], 16)


def run_condition(ex: Example, policy: str, schedule: str, rep: int,
                  execute, template, count, budget: Budget, donor=None) -> dict:
    """Run one (question, policy, schedule, repetition) and return a log record.
    `donor` ({agent: Report}) supplies the irrelevant-peer control's reports."""
    shards = dict(zip("AB", shard(ex)))
    store = ReportStore()
    captured: dict[str, Captured] = {}
    pending: dict[str, object] = {}
    calls = []
    trace = []
    cond = f"{ex.qid}:{policy}:{schedule}:{rep}"

    for op, agent in SCHEDULES[schedule]:
        trace.append(f"{op}:{agent}")
        if op == "capture":
            peer, vis = peer_section(policy, store, agent, donor)
            msgs = prompts.analyst_messages(ex, agent, shards[agent], peer)
            captured[agent] = Captured(agent, template(msgs), vis, peer, store.published())
        elif op == "infer":
            c = captured[agent]
            seed = seed_for(ex.qid, agent, rep)
            res, cached = execute(c.prompt, budget.analyst_out, seed)
            rep_obj, diag = parse_report(agent, res.text, report_id=f"{cond}:{agent}")
            pending[agent] = rep_obj
            calls.append({"role": f"analyst_{agent}", "prompt": c.prompt,
                          "prompt_hash": hashlib.sha256(c.prompt.encode()).hexdigest(),
                          "visible_report_ids": list(c.visible_report_ids),
                          "captured_after": list(c.captured_after),
                          "peer_visible": bool(c.visible_report_ids),
                          "output": res.text, "in_tokens": res.in_tokens,
                          "out_tokens": res.out_tokens, "seconds": res.seconds,
                          "cached": cached, "seed": seed,
                          "over_budget": res.in_tokens > budget.max_input,
                          "parsed": {"findings": [[f.text, list(f.ids)] for f in rep_obj.findings],
                                     "answer": rep_obj.answer}, **diag})
        elif op == "publish":
            store.publish(agent, pending.pop(agent))

    # Orchestrator: always both complete reports, canonical A-then-B order.
    ra, rb = store.get("A"), store.get("B")
    msgs = prompts.orchestrator_messages(ex, render_full(ra), render_full(rb))
    prompt = template(msgs)
    seed = seed_for(ex.qid, "orch", rep)
    res, cached = execute(prompt, budget.final_out, seed)
    answer, support = parse_final(res.text)
    calls.append({"role": "orchestrator", "prompt": prompt,
                  "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                  "visible_report_ids": [ra.report_id, rb.report_id],
                  "output": res.text, "in_tokens": res.in_tokens, "out_tokens": res.out_tokens,
                  "seconds": res.seconds, "cached": cached, "seed": seed,
                  "over_budget": res.in_tokens > budget.max_input,
                  "parsed": {"answer": answer, "support": list(support)},
                  "valid": answer is not None})
    return {"dataset": ex.dataset, "qid": ex.qid, "policy": policy, "schedule": schedule,
            "rep": rep, "trace": trace, "store_events": store.events,
            "final_answer": answer, "final_support": list(support),
            "analyst_answers": {"A": ra.answer, "B": rb.answer}, "calls": calls}


def no_peer_reports(ex: Example, execute, template, budget: Budget) -> dict:
    """Each analyst's report with no peer visible (the Both-captured history),
    used as donor material for the irrelevant-peer control."""
    rec = run_condition(ex, "isolation", "both_captured", 0, execute, template, None, budget)
    out = {}
    for c in rec["calls"]:
        if c["role"].startswith("analyst_"):
            out[c["role"][-1]] = parse_report(c["role"][-1], c["output"],
                                              report_id=f"{ex.qid}:donor:{c['role'][-1]}")[0]
    return out


def run_single(ex: Example, rep: int, execute, template, budget: Budget) -> dict:
    """Single-agent reference: one call over all evidence."""
    prompt = template(prompts.single_messages(ex))
    seed = seed_for(ex.qid, "single", rep)
    res, cached = execute(prompt, budget.final_out, seed)
    answer, support = parse_final(res.text)
    return {"dataset": ex.dataset, "qid": ex.qid, "policy": "single_agent", "schedule": "single",
            "rep": rep, "final_answer": answer, "final_support": list(support),
            "calls": [{"role": "single", "prompt": prompt,
                       "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                       "output": res.text, "in_tokens": res.in_tokens, "out_tokens": res.out_tokens,
                       "seconds": res.seconds, "cached": cached, "seed": seed,
                       "over_budget": res.in_tokens > budget.single_max_input,
                       "parsed": {"answer": answer, "support": list(support)},
                       "valid": answer is not None}]}
