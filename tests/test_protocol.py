import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schedsens.data import Example, Paragraph, shard
from schedsens.evaluate import score, support_units
from schedsens.llm import Executor, GenResult, ResponseCache
from schedsens.reports import (NO_PEER, Finding, Report, ReportStore, parse_final,
                               parse_report, peer_section)
from schedsens.scheduler import SCHEDULES, Budget, run_condition


def hotpot_ex(qid="q1"):
    paras = tuple(Paragraph(i, f"T{i}", (f"s0 of {i}.", f"s1 of {i}.")) for i in range(10))
    return Example("hotpotqa", qid, "Who?", paras, "Alice", (), (("T2", 1), ("T7", 0)))


def musique_ex(qid="m1"):
    paras = tuple(Paragraph(i, f"T{i}", (f"text {i}",)) for i in range(20))
    return Example("musique", qid, "Where?", paras, "Paris", ("City of Light",), (3, 11))


# ------------------------------------------------------------- sharding

def test_shard_disjoint_cover_ordered_deterministic():
    ex = hotpot_ex()
    a, b = shard(ex)
    ia, ib = [p.pid for p in a], [p.pid for p in b]
    assert len(ia) == len(ib) == 5
    assert set(ia).isdisjoint(ib) and set(ia) | set(ib) == set(range(10))
    assert ia == sorted(ia) and ib == sorted(ib)
    assert shard(ex) == (a, b)
    assert shard(hotpot_ex("other")) != (a, b)  # depends on qid (overwhelmingly likely)


def test_shard_ignores_gold():
    ex = hotpot_ex()
    ex2 = Example(ex.dataset, ex.qid, ex.question, ex.paragraphs, "Bob", (), (("T0", 0),))
    assert shard(ex) == shard(ex2)


# ------------------------------------------------------------- store

def test_store_write_once_owner_only():
    s = ReportStore()
    r = Report("A", (), "x")
    with pytest.raises(PermissionError):
        s.publish("B", r)
    s.publish("A", r)
    with pytest.raises(RuntimeError):
        s.publish("A", r)


# ------------------------------------------------------------- visibility

def _store_with_b():
    s = ReportStore()
    s.publish("B", Report("B", (Finding("Alice wrote it", ("P2.1",)),), "Alice", "rid-B"))
    return s


def test_visibility_policies():
    s = _store_with_b()
    full, vis = peer_section("full", s, "A")
    assert "Alice wrote it" in full and "Provisional answer: Alice" in full and vis == ("rid-B",)
    ev, vis = peer_section("evidence_only", s, "A")
    assert "Alice wrote it" in ev and "Provisional answer" not in ev and "P2.1" in ev
    iso, vis = peer_section("isolation", s, "A")
    assert iso == NO_PEER and vis == ()
    # analyst never sees its own report as "peer"
    assert peer_section("full", s, "B") == (NO_PEER, ())


# ------------------------------------------------------------- parsing

def test_parse_report():
    txt = ("FINDING 1: X founded Y. [IDs: P3.1, P3.2]\n"
           "**Finding 2:** Y is in Z [IDs: P5]\n"
           "Finding 3: extra [IDs: P1.0]\n"
           "ANSWER: \"Zurich\".")
    r, d = parse_report("A", txt)
    assert r.answer == "Zurich" and len(r.findings) == 2 and d["dropped_findings"] == 1
    assert r.findings[0].ids == ("P3.1", "P3.2") and r.findings[1].ids == ("P5",)
    r, d = parse_report("A", "I think it is Zurich")
    assert not r.valid and d["valid"] is False
    r, d = parse_report("A", "FINDING 1: nothing cited\nANSWER: insufficient evidence")
    assert d["findings_missing_ids"] == 1


def test_parse_final():
    assert parse_final("ANSWER: yes\nSUPPORT: P1.0, P4.2") == ("yes", ("P1.0", "P4.2"))
    assert parse_final("no format") == (None, ())


# ------------------------------------------------------------- scoring

def test_score_hotpot_and_missing():
    ex = hotpot_ex()
    s = score(ex, "alice", ["P2.1", "P7.0"])
    assert s["em"] == 1 and s["sp_em"] == 1 and s["joint_em"] == 1
    s = score(ex, None, [])
    assert s["em"] == 0 and s["f1"] == 0 and s["missing_answer"]
    units, bad = support_units(ex, ["P2", "P99.0", "P2.5", "P2.1"])
    assert units == [["T2", 1]] and bad == 3


def test_score_musique_aliases_and_support():
    ex = musique_ex()
    s = score(ex, "city of light", ["P3", "P11.0"])
    assert s["em"] == 1 and s["sp_em"] == 1


# ------------------------------------------------------------- scheduler

class FakeModel:
    """Answers depend on whether a peer report is visible, so visibility
    leaks would show up in outputs."""
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_new_tokens, seed):
        self.prompts.append(prompt)
        if "orchestrator" in prompt or "=== Peer report section ===" not in prompt:
            return GenResult("ANSWER: Alice\nSUPPORT: P2.1", len(prompt), 5, 0.0)
        tag = "peer" if "Report from Analyst" in prompt.split("=== Peer report section ===")[1] else "solo"
        return GenResult(f"FINDING 1: saw {tag} [IDs: P2.1]\nANSWER: {tag}", len(prompt), 9, 0.0)


def _run(policy, schedule, reuse=False):
    m = FakeModel()
    execu = Executor(m, ResponseCache(None), reuse)
    rec = run_condition(hotpot_ex(), policy, schedule, 0, execu, lambda msgs: "\n".join(
        x["content"] for x in msgs), len, Budget())
    return rec, m


@pytest.mark.parametrize("policy", ["full", "evidence_only", "isolation"])
def test_schedules_capture_order(policy):
    rec, _ = _run(policy, "a_first")
    calls = {c["role"]: c for c in rec["calls"]}
    assert calls["analyst_A"]["peer_visible"] is False
    assert calls["analyst_B"]["peer_visible"] is (policy != "isolation")
    assert rec["analyst_answers"]["A"] == "solo"
    assert rec["analyst_answers"]["B"] == ("solo" if policy == "isolation" else "peer")

    rec, _ = _run(policy, "b_first")
    calls = {c["role"]: c for c in rec["calls"]}
    assert calls["analyst_B"]["peer_visible"] is False
    assert calls["analyst_A"]["peer_visible"] is (policy != "isolation")

    rec, _ = _run(policy, "both_captured")
    assert all(not c["peer_visible"] for c in rec["calls"] if c["role"].startswith("analyst"))
    assert rec["trace"][:2] == ["capture:A", "capture:B"]


def test_evidence_only_hides_answer_in_prompt():
    rec, _ = _run("evidence_only", "a_first")
    pb = next(c for c in rec["calls"] if c["role"] == "analyst_B")["prompt"]
    peer = pb.split("=== Peer report section ===")[1]
    assert "saw solo" in peer and "Provisional answer" not in peer


def test_orchestrator_order_canonical():
    rec, _ = _run("full", "b_first")
    p = rec["calls"][-1]["prompt"]
    assert p.index("Report from Analyst A") < p.index("Report from Analyst B")
    assert rec["calls"][-1]["visible_report_ids"][0].endswith(":A")


def test_capture_is_immutable_after_publish():
    # both_captured: B's prompt was captured before A published, so it must not
    # contain A's report even though A is published before B infers.
    rec, m = _run("full", "both_captured")
    pb = next(c for c in rec["calls"] if c["role"] == "analyst_B")["prompt"]
    assert NO_PEER in pb and "saw solo" not in pb


def test_reuse_counts_cached_calls():
    m = FakeModel()
    execu = Executor(m, ResponseCache(None), reuse=True)
    tmpl = lambda msgs: "\n".join(x["content"] for x in msgs)
    for pol in ("full", "evidence_only", "isolation"):
        run_condition(hotpot_ex(), pol, "both_captured", 0, execu, tmpl, len, Budget())
    assert len(m.prompts) == 3  # identical prompts executed once
    execu = Executor(FakeModel(), ResponseCache(None), reuse=False)
    rec = run_condition(hotpot_ex(), "full", "both_captured", 0, execu, tmpl, len, Budget())
    assert not any(c["cached"] for c in rec["calls"])


# ------------------------------------------------------------- extensions

from schedsens.scheduler import no_peer_reports, run_single


def wiki_ex(qid="w1"):
    paras = tuple(Paragraph(i, f"T{i}", (f"s0 of {i}.", f"s1 of {i}.")) for i in range(10))
    return Example("2wiki", qid, "Who?", paras, "Alice Smith", ("Alice",), (("T1", 0), ("T4", 1)))


def test_score_2wiki_aliases_and_support():
    ex = wiki_ex()
    s = score(ex, "alice", ["P1.0", "P4.1"])
    assert s["em"] == 1 and s["sp_em"] == 1 and s["joint_em"] == 1
    assert score(ex, "Bob", [])["em"] == 0


def test_irrelevant_control_timing_and_content():
    tmpl = lambda msgs: "\n".join(x["content"] for x in msgs)
    m = FakeModel()
    execu = Executor(m, ResponseCache(None), reuse=False)
    donor = {"A": Report("A", (Finding("UNRELATED FACT", ("P9.9",)),), "Zed", "donor:A"),
             "B": Report("B", (Finding("OTHER FACT", ("P8.8",)),), "Yan", "donor:B")}
    rec = run_condition(hotpot_ex(), "irrelevant", "a_first", 0, execu, tmpl, len, Budget(), donor)
    calls = {c["role"]: c for c in rec["calls"]}
    assert calls["analyst_A"]["peer_visible"] is False          # same timing as full sharing
    pb = calls["analyst_B"]["prompt"].split("=== Peer report section ===")[1]
    assert "UNRELATED FACT" in pb and "Provisional answer: Zed" in pb and "saw solo" not in pb
    assert calls["analyst_B"]["visible_report_ids"] == ["donor:A"]
    # orchestrator still receives the real reports
    assert "saw" in rec["calls"][-1]["prompt"]
    rec = run_condition(hotpot_ex(), "irrelevant", "both_captured", 0, execu, tmpl, len, Budget(), donor)
    assert all(not c["peer_visible"] for c in rec["calls"] if c["role"].startswith("analyst"))


def test_no_peer_reports_and_single():
    tmpl = lambda msgs: "\n".join(x["content"] for x in msgs)
    execu = Executor(FakeModel(), ResponseCache(None), reuse=False)
    d = no_peer_reports(hotpot_ex(), execu, tmpl, Budget())
    assert set(d) == {"A", "B"} and d["A"].answer == "solo"
    rec = run_single(hotpot_ex(), 0, execu, tmpl, Budget())
    assert rec["policy"] == "single_agent" and len(rec["calls"]) == 1
    assert "s1 of 9." in rec["calls"][0]["prompt"]                 # sees all evidence


def test_wiki_prompt_uses_sentence_ids():
    from schedsens import prompts
    a, _ = shard(wiki_ex())
    txt = prompts.render_evidence(wiki_ex(), a)
    assert f"[P{a[0].pid}.0]" in txt
