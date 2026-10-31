# Schedule Sensitivity in Shared-State Multi-Agent LLM Systems: Separating Peer Presence from Peer Content

Controlled measurement of how context-capture order and peer-report visibility
affect a two-analyst + orchestrator workflow. Two models (Llama-3-8B-Instruct and
Qwen2.5-7B-Instruct, both 4-bit) run on three multi-hop QA development subsets:
HotpotQA distractor (500), MuSiQue-Ans (300) and 2WikiMultiHopQA (300).
Conditions:

- Three visibility policies: full, evidence-only and isolation.
- An irrelevant-peer control.
- Three logical schedules.
- A single-agent reference.
- Identical-condition repeats.

## Layout

| Path | Contents |
|---|---|
| `schedsens/data.py` | Dataset adapters (HotpotQA, MuSiQue, 2Wiki), question-ID-seeded sharding, structural checks |
| `schedsens/reports.py` | Report type, write-once owner-only store, visibility policies (incl. irrelevant-peer control), parsers |
| `schedsens/prompts.py` | Analyst, orchestrator and single-agent templates (fixed across conditions) |
| `schedsens/scheduler.py` | Logical schedules (A-first, B-first, both-captured), single-agent run |
| `schedsens/llm.py` | 4-bit HF inference wrapper (chunked prefill for long prompts), prompt-hash response cache |
| `schedsens/evaluate.py` | Per-question scoring with the vendored official evaluators |
| `scripts/build_manifest.py` | Freezes seeded question manifests and exclusions |
| `scripts/run.py` | Resumable JSONL runner (`--mode workflow|single`, `--policies`, `--no-reuse`) |
| `scripts/run_queue.sh` | Runs every stage for both models in order |
| `scripts/analyze.py` | Paired question-level bootstrap analysis -> `results/summary.json` |
| `scripts/make_tables.py`, `scripts/figures.py` | LaTeX tables and figure, generated from the summary (written to `paper/`) |
| `scripts/verify_repro.py` | Re-executes logged conditions and checks byte-identical outputs |
| `third_party/` | Official HotpotQA, MuSiQue and 2Wiki metric code (unchanged, with licenses) |

## Reproduce

Run logs are stored gzip-compressed (`runs/<model>/*.jsonl.gz`); `scripts/analyze.py`
reads them directly, so the paper's tables and figure can be regenerated without a GPU:

```bash
.venv/Scripts/python scripts/analyze.py && .venv/Scripts/python scripts/make_tables.py && .venv/Scripts/python scripts/figures.py
```

Full re-run from scratch:

```bash
git clone https://github.com/atishay16/schedule_sensitivity
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests
.venv/Scripts/python scripts/build_manifest.py
bash scripts/run_queue.sh main          # about 37 h on one 8 GB RTX 3070 laptop GPU
.venv/Scripts/python scripts/analyze.py
.venv/Scripts/python scripts/make_tables.py
.venv/Scripts/python scripts/figures.py
```

HotpotQA and MuSiQue are fetched from the Hugging Face mirrors
(`hotpotqa/hotpot_qa`, distractor config; `dgslibisey/MuSiQue`). 2WikiMultiHopQA
comes from the official release, including `id_aliases.json`. Checksums are in
`data/raw/SHA256SUMS`.

Results are on adapted, length-filtered development subsets, not official
leaderboard submissions.
