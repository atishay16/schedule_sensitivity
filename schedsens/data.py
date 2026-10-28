"""Dataset adapters for HotpotQA (distractor), MuSiQue-Ans and 2WikiMultiHopQA.

Each example is converted to a common `Example` whose evidence units carry
original source IDs.  Gold answers and support labels are kept on the
Example for scoring only; nothing in `prompts.py` reads them.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


@dataclass(frozen=True)
class Paragraph:
    pid: int                      # original paragraph index in the dataset record
    title: str
    sentences: tuple[str, ...]    # HotpotQA: sentence list; MuSiQue: single text


@dataclass(frozen=True)
class Example:
    dataset: str                  # "hotpotqa" | "musique"
    qid: str
    question: str
    paragraphs: tuple[Paragraph, ...]
    # --- gold, scoring only ---
    answer: str
    answer_aliases: tuple[str, ...] = ()
    gold_support: tuple = ()      # hotpot: ((title, sent_id), ...); musique: (pid, ...)
    meta: dict = field(default_factory=dict, compare=False, hash=False)


# ---------------------------------------------------------------- HotpotQA

def _hotpot_row(row: dict) -> Example:
    ctx = row["context"]
    paras = tuple(
        Paragraph(i, t, tuple(s.strip() for s in sents))
        for i, (t, sents) in enumerate(zip(ctx["title"], ctx["sentences"]))
    )
    sf = row["supporting_facts"]
    gold = tuple((t, int(s)) for t, s in zip(sf["title"], sf["sent_id"]))
    return Example("hotpotqa", row["id"], row["question"], paras, row["answer"], (),
                   gold, {"type": row.get("type"), "level": row.get("level")})


def load_hotpot(path: str | Path) -> Iterator[Example]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            yield _hotpot_row(row)


# ---------------------------------------------------------------- MuSiQue

def _musique_row(row: dict) -> Example:
    paras = tuple(
        Paragraph(int(p["idx"]), p["title"], (p["paragraph_text"].strip(),))
        for p in row["paragraphs"]
    )
    gold = tuple(int(p["idx"]) for p in row["paragraphs"] if p["is_supporting"])
    return Example("musique", row["id"], row["question"], paras, row["answer"],
                   tuple(row.get("answer_aliases", [])), gold,
                   {"answerable": row.get("answerable", True),
                    "hops": len(row.get("question_decomposition", []))})


def load_musique(path: str | Path) -> Iterator[Example]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield _musique_row(json.loads(line))


# ---------------------------------------------------------------- 2WikiMultiHopQA

def _iter_json_array(path: str | Path, chunk: int = 1 << 20) -> Iterator[dict]:
    """Stream objects from a top-level JSON array without loading it whole."""
    dec = json.JSONDecoder()
    buf, pos = "", 0
    with open(path, encoding="utf-8") as f:
        buf = f.read(chunk)
        pos = buf.index("[") + 1
        while True:
            while pos < len(buf) and buf[pos] in " \r\n\t,":
                pos += 1
            if pos < len(buf) and buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            yield obj
            pos = end
            if pos > chunk:
                buf, pos = buf[pos:], 0


def _load_2wiki_aliases(path: Path) -> dict[str, tuple[str, ...]]:
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["Q_id"]] = tuple(r["aliases"] + r["demonyms"])
    return out


def load_2wiki(path: str | Path) -> Iterator[Example]:
    """2WikiMultiHopQA (data_ids release); aliases from id_aliases.json in the
    same folder, as used by the official v1.1 evaluator."""
    path = Path(path)
    aliases = _load_2wiki_aliases(path.parent / "id_aliases.json")
    for row in _iter_json_array(path):
        paras = tuple(Paragraph(i, t, tuple(s.strip() for s in sents))
                      for i, (t, sents) in enumerate(row["context"]))
        gold = tuple((t, int(s)) for t, s in row["supporting_facts"])
        yield Example("2wiki", row["_id"], row["question"], paras, row["answer"],
                      aliases.get(row.get("answer_id"), ()), gold, {"type": row.get("type")})


def load(dataset: str, path: str | Path) -> Iterator[Example]:
    return {"hotpotqa": load_hotpot, "musique": load_musique, "2wiki": load_2wiki}[dataset](path)


# ---------------------------------------------------------------- sharding

def _qid_rng(qid: str, salt: str) -> random.Random:
    h = hashlib.sha256(f"{salt}:{qid}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def shard(ex: Example, salt: str = "shard-v1") -> tuple[tuple[Paragraph, ...], tuple[Paragraph, ...]]:
    """Split paragraphs into two equal-count shards (A, B) with a
    question-ID-derived shuffle; original order is preserved within a shard.
    Uses no gold labels."""
    idx = list(range(len(ex.paragraphs)))
    _qid_rng(ex.qid, salt).shuffle(idx)
    half = len(idx) // 2
    a, b = sorted(idx[:half]), sorted(idx[half:])
    return tuple(ex.paragraphs[i] for i in a), tuple(ex.paragraphs[i] for i in b)


def structural_check(ex: Example) -> str | None:
    """Return a reason string if the example is structurally invalid."""
    n = len(ex.paragraphs)
    if ex.dataset in ("hotpotqa", "2wiki") and n != 10:
        return f"paragraph_count={n}"
    if ex.dataset == "musique":
        if n != 20:
            return f"paragraph_count={n}"
        if not ex.meta.get("answerable", True):
            return "unanswerable"
    if n % 2:
        return "odd_paragraph_count"
    if any(len(p.sentences) == 0 for p in ex.paragraphs):
        return "empty_paragraph"
    if not ex.question.strip() or not ex.answer.strip():
        return "empty_question_or_answer"
    return None
