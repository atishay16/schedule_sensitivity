"""Structured analyst reports, a write-once agent-owned report store, and
the three visibility policies.

The store intentionally excludes lost updates and write-write races: each
agent may publish exactly one report, only under its own name.  The study
varies *visibility*, not corruption of shared state.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

POLICIES = ("full", "evidence_only", "isolation")
AGENTS = ("A", "B")


@dataclass(frozen=True)
class Finding:
    text: str
    ids: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    agent: str
    findings: tuple[Finding, ...]
    answer: str | None            # None => no ANSWER line (invalid report)
    report_id: str = ""

    @property
    def valid(self) -> bool:
        return self.answer is not None

    def digest(self) -> str:
        return hashlib.sha256(render_full(self).encode()).hexdigest()[:12]


class ReportStore:
    """Write-once, owner-only store.  Reads return immutable Report objects."""

    def __init__(self) -> None:
        self._reports: dict[str, Report] = {}
        self.events: list[tuple[str, str]] = []   # (event, agent) trace

    def publish(self, writer: str, report: Report) -> None:
        if writer != report.agent:
            raise PermissionError(f"{writer} may not write report owned by {report.agent}")
        if report.agent in self._reports:
            raise RuntimeError(f"report for {report.agent} already published")
        self._reports[report.agent] = report
        self.events.append(("publish", writer))

    def get(self, agent: str) -> Report | None:
        return self._reports.get(agent)

    def published(self) -> tuple[str, ...]:
        return tuple(self._reports)


def peer_of(agent: str) -> str:
    return "B" if agent == "A" else "A"


# ------------------------------------------------------------- rendering

_EMPTY = "(report missing or unparseable)"


def _render_findings(r: Report) -> list[str]:
    lines = []
    for i, f in enumerate(r.findings, 1):
        ids = ", ".join(f.ids) if f.ids else "none"
        lines.append(f"Finding {i}: {f.text} [IDs: {ids}]")
    return lines


def render_full(r: Report | None) -> str:
    if r is None or (not r.findings and r.answer is None):
        return _EMPTY
    lines = _render_findings(r) or ["(no findings)"]
    lines.append(f"Provisional answer: {r.answer if r.answer is not None else '(none)'}")
    return "\n".join(lines)


def render_evidence_only(r: Report | None) -> str:
    if r is None or (not r.findings and r.answer is None):
        return _EMPTY
    return "\n".join(_render_findings(r) or ["(no findings)"])


NO_PEER = "No peer report is visible to you."


# Control policy: same timing as full sharing, but the visible report is the
# same-role no-peer report written for a *different* question (format- and
# length-matched, content irrelevant).  Separates peer content from the mere
# presence of a peer section.
CONTROL_POLICIES = ("irrelevant",)


def peer_section(policy: str, store: ReportStore, agent: str,
                 donor: dict[str, Report] | None = None) -> tuple[str, tuple[str, ...]]:
    """Return (peer section text, visible report IDs) for `agent` reading the
    store *now* under `policy`.  Called exactly once, at prompt capture."""
    if policy not in POLICIES + CONTROL_POLICIES:
        raise ValueError(policy)
    peer = store.get(peer_of(agent))
    if policy == "isolation" or peer is None:
        return NO_PEER, ()
    if policy == "irrelevant":
        d = donor[peer.agent]
        return f"Report from Analyst {peer.agent}:\n{render_full(d)}", (d.report_id,)
    body = render_full(peer) if policy == "full" else render_evidence_only(peer)
    return f"Report from Analyst {peer.agent}:\n{body}", (peer.report_id,)


# ------------------------------------------------------------- parsing

_FIND_RE = re.compile(r"^\s*\**\s*finding\s*(\d+)\s*\**\s*[:.\-]\s*(.*)$", re.I)
_ANS_RE = re.compile(r"^\s*\**\s*(?:provisional\s+|final\s+)?answer\s*\**\s*[:\-]\s*(.*)$", re.I)
_SUP_RE = re.compile(r"^\s*\**\s*support(?:ing)?(?:\s+ids)?\s*\**\s*[:\-]\s*(.*)$", re.I)
_IDS_BLOCK = re.compile(r"\[\s*IDs?\s*:\s*([^\]]*)\]", re.I)
ID_RE = re.compile(r"\bP(\d+)(?:\.(\d+))?\b")


def extract_ids(s: str) -> tuple[str, ...]:
    out = []
    for m in ID_RE.finditer(s):
        tok = f"P{m.group(1)}" + (f".{m.group(2)}" if m.group(2) is not None else "")
        if tok not in out:
            out.append(tok)
    return tuple(out)


def _clean_answer(s: str) -> str:
    s = s.strip().strip("*").strip()
    s = re.sub(r"\s*\[IDs?:[^\]]*\]\s*$", "", s, flags=re.I)
    s = s.rstrip(".").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    return s.strip().rstrip(".").strip()


MAX_FINDINGS = 2


def parse_report(agent: str, text: str, report_id: str = "") -> tuple[Report, dict]:
    """Deterministic analyst-report parser.  Returns the report plus
    diagnostics (extra findings dropped, findings lacking IDs)."""
    findings: list[Finding] = []
    answer = None
    dropped = 0
    for line in text.splitlines():
        if (m := _FIND_RE.match(line)):
            body = m.group(2)
            ids_m = _IDS_BLOCK.search(body)
            ids = extract_ids(ids_m.group(1)) if ids_m else extract_ids(body)
            txt = _IDS_BLOCK.sub("", body).strip()
            if len(findings) < MAX_FINDINGS:
                findings.append(Finding(txt, ids))
            else:
                dropped += 1
        elif answer is None and (m := _ANS_RE.match(line)):
            answer = _clean_answer(m.group(1))
    rep = Report(agent, tuple(findings), answer, report_id)
    diag = {"dropped_findings": dropped,
            "findings_missing_ids": sum(1 for f in findings if not f.ids),
            "valid": rep.valid}
    return rep, diag


def parse_final(text: str) -> tuple[str | None, tuple[str, ...]]:
    answer, support = None, ()
    for line in text.splitlines():
        if answer is None and (m := _ANS_RE.match(line)):
            answer = _clean_answer(m.group(1))
        elif not support and (m := _SUP_RE.match(line)):
            support = extract_ids(m.group(1))
    return answer, support
