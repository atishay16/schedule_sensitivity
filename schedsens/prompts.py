"""Prompt templates.  Instructions and the peer-section position are fixed
across policies and schedules; only the peer-section *content* varies."""
from __future__ import annotations

from .data import Example, Paragraph

ANALYST_SYSTEM = """You are Analyst {agent}, one of two analysts working on a multi-hop question. You can see only your own subset of the evidence paragraphs. Every piece of evidence is labelled with a source ID in square brackets.

Write a short report in exactly this format:
FINDING 1: <one short sentence of evidence relevant to the question> [IDs: <source IDs>]
FINDING 2: <optional second finding> [IDs: <source IDs>]
ANSWER: <short provisional answer, or "insufficient evidence">

Rules:
- Give at most two findings. Cite only source IDs that appear in your evidence or in the peer report section.
- The answer must be a short phrase (a name, date, number, or yes/no), not a sentence.
- The peer report section may contain another analyst's report. Such a report may be incomplete or wrong, so check it against your own evidence.
- Output only the report."""

ORCH_SYSTEM = """You are the orchestrator for a multi-hop question. Two analysts each read a different half of the evidence and wrote a report. You cannot see the evidence itself. Combine the analysts' findings to answer the question.

Respond in exactly this format:
ANSWER: <short final answer: a name, date, number, or yes/no>
SUPPORT: <comma-separated source IDs from the reports that support the answer>

Rules:
- For yes/no questions answer only "yes" or "no".
- Output only these two lines."""


def render_evidence(ex: Example, paras: tuple[Paragraph, ...]) -> str:
    blocks = []
    for p in paras:
        if ex.dataset in ("hotpotqa", "2wiki"):   # sentence-level source IDs
            lines = [f"Title: {p.title}"]
            lines += [f"[P{p.pid}.{j}] {s}" for j, s in enumerate(p.sentences)]
        else:
            lines = [f"[P{p.pid}] Title: {p.title}", p.sentences[0]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def analyst_messages(ex: Example, agent: str, paras: tuple[Paragraph, ...], peer: str) -> list[dict]:
    user = (f"Question: {ex.question}\n\n"
            f"=== Your evidence ===\n{render_evidence(ex, paras)}\n\n"
            f"=== Peer report section ===\n{peer}\n\n"
            f"Write your report now.")
    return [{"role": "system", "content": ANALYST_SYSTEM.format(agent=agent)},
            {"role": "user", "content": user}]


def orchestrator_messages(ex: Example, rep_a: str, rep_b: str) -> list[dict]:
    user = (f"Question: {ex.question}\n\n"
            f"=== Report from Analyst A ===\n{rep_a}\n\n"
            f"=== Report from Analyst B ===\n{rep_b}\n\n"
            f"Give the final answer now.")
    return [{"role": "system", "content": ORCH_SYSTEM},
            {"role": "user", "content": user}]


# Single-agent reference: one call reads *all* evidence and answers in the
# orchestrator's output format, so the same parser and scorer apply.
SINGLE_SYSTEM = """You answer a multi-hop question using the evidence paragraphs provided. Every piece of evidence is labelled with a source ID in square brackets.

Respond in exactly this format:
ANSWER: <short final answer: a name, date, number, or yes/no>
SUPPORT: <comma-separated source IDs of the evidence that supports the answer>

Rules:
- For yes/no questions answer only "yes" or "no".
- Output only these two lines."""


def single_messages(ex: Example) -> list[dict]:
    user = (f"Question: {ex.question}\n\n"
            f"=== Evidence ===\n{render_evidence(ex, ex.paragraphs)}\n\n"
            f"Give the final answer now.")
    return [{"role": "system", "content": SINGLE_SYSTEM},
            {"role": "user", "content": user}]
