#!/usr/bin/env bash
# Runs every experiment stage in order. Each step resumes from its JSONL log,
# so re-running this script after an interruption continues where it stopped.
#   bash scripts/run_queue.sh pilot   # pilots for new conditions and models
#   bash scripts/run_queue.sh main    # full study
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
mkdir -p runs/logs
STATUS=runs/logs/queue_status.txt

step() {  # step <name> <args...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$STATUS"
  if $PY scripts/run.py "$@" > "runs/logs/$name.log" 2>&1; then
    echo "$(date '+%F %T') DONE  $name" >> "$STATUS"
  else
    echo "$(date '+%F %T') FAIL  $name (see runs/logs/$name.log)" >> "$STATUS"
  fi
}

phase=${1:-main}
MODELS="llama3-8b qwen2.5-7b"
DATASETS="hotpotqa musique 2wiki"

if [ "$phase" = pilot ]; then
  for m in $MODELS; do
    for d in $DATASETS; do
      step "pilot_${m}_${d}"            --model $m --stage pilot_$d
      step "pilot_${m}_${d}_irrelevant" --model $m --stage pilot_$d --policies irrelevant
      step "pilot_${m}_${d}_single"     --model $m --stage pilot_$d --mode single
    done
  done
fi

if [ "$phase" = main ]; then
  for m in $MODELS; do
    for d in $DATASETS; do
      step "main_${m}_${d}" --model $m --stage main_$d
    done
    for d in $DATASETS; do
      step "main_${m}_${d}_single"     --model $m --stage main_$d --mode single
      step "main_${m}_${d}_irrelevant" --model $m --stage main_$d --policies irrelevant
      step "repeat_${m}_${d}"          --model $m --stage repeat_$d --policies full --rep 1 --no-reuse
    done
  done
fi
echo "$(date '+%F %T') QUEUE $phase FINISHED" >> "$STATUS"
