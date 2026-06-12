#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_exp5b_50agent.sh [DATA_DIR] [OUTPUT_ROOT]

Runs exp5b on 50-agent data:
  - numeric: SenFeedTruth, DecentTruth, SenteTruth, BasicTruth, LASOTruth
  - text:    BasicTruth, LASOTruth, SenteTruth

DATA_DIR may contain num_demo.json/demo.json directly, or contain a
data_agent-50_news-300/ subdirectory with those files.

Environment overrides:
  AGENT_COUNTS              default: 50
  NUM_BATCH_GROUP_SIZE      default: 10
  NUM_WORKLOADS             default: 3
  NUM_SUBSET_SAMPLES        default: 2
  TEXT_BATCH_GROUP_SIZE     default: 5
  TEXT_WORKLOADS            default: 1
  TEXT_SUBSET_SAMPLES       default: 1
  WORKER_STARTUP_TIMEOUT    default: 180
  WORKER_SPAWN_STAGGER      default: 0.5
  NUMERIC_PORT              default: 5560
  TEXT_PORT                 default: 5588
  PYTHON_BIN                default: python3
  RUN_DOMAIN                default: both (both|numeric|text)
  TRUTHFINDER_SBERT_MODEL    default: sentence-transformers/all-MiniLM-L6-v2
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${1:-${ROOT_DIR}/data}"
OUTPUT_ROOT="${2:-${ROOT_DIR}/decentralized_truth_discovery/examples/outputs}"

if [[ -f "${DATA_DIR}/num_demo.json" && -f "${DATA_DIR}/demo.json" ]]; then
  NUM_JSON="${DATA_DIR}/num_demo.json"
  TEXT_JSON="${DATA_DIR}/demo.json"
elif [[ -f "${DATA_DIR}/data_agent-50_news-300/num_demo.json" && -f "${DATA_DIR}/data_agent-50_news-300/demo.json" ]]; then
  NUM_JSON="${DATA_DIR}/data_agent-50_news-300/num_demo.json"
  TEXT_JSON="${DATA_DIR}/data_agent-50_news-300/demo.json"
else
  echo "Could not find num_demo.json and demo.json under: ${DATA_DIR}" >&2
  usage >&2
  exit 2
fi

AGENT_COUNTS="${AGENT_COUNTS:-50}"
NUM_BATCH_GROUP_SIZE="${NUM_BATCH_GROUP_SIZE:-10}"
NUM_WORKLOADS="${NUM_WORKLOADS:-3}"
NUM_SUBSET_SAMPLES="${NUM_SUBSET_SAMPLES:-2}"
TEXT_BATCH_GROUP_SIZE="${TEXT_BATCH_GROUP_SIZE:-5}"
TEXT_WORKLOADS="${TEXT_WORKLOADS:-1}"
TEXT_SUBSET_SAMPLES="${TEXT_SUBSET_SAMPLES:-1}"
WORKER_STARTUP_TIMEOUT="${WORKER_STARTUP_TIMEOUT:-180}"
WORKER_SPAWN_STAGGER="${WORKER_SPAWN_STAGGER:-0.5}"
NUMERIC_PORT="${NUMERIC_PORT:-5560}"
TEXT_PORT="${TEXT_PORT:-5588}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_DOMAIN="${RUN_DOMAIN:-both}"
export TRUTHFINDER_SBERT_MODEL="${TRUTHFINDER_SBERT_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${ROOT_DIR}"

echo "Data numeric: ${NUM_JSON}"
echo "Data text   : ${TEXT_JSON}"
echo "Output root : ${OUTPUT_ROOT}"
echo "SBERT model : ${TRUTHFINDER_SBERT_MODEL}"
echo "Python      : ${PYTHON_BIN}"
echo "Run domain  : ${RUN_DOMAIN}"

run_numeric_method() {
  local method="$1"
  local exp_id="exp5b_numeric_dist_50agent_${method}"
  "${PYTHON_BIN}" decentralized_truth_discovery/examples/agent_scalability_distributed.py \
    --mode dist-tcp \
    --coord-addr "tcp://127.0.0.1:${NUMERIC_PORT}" \
    --input-json "${NUM_JSON}" \
    --output-root "${OUTPUT_ROOT}" \
    --experiment-id "${exp_id}" \
    --agent-counts "${AGENT_COUNTS}" \
    --subset-samples "${NUM_SUBSET_SAMPLES}" \
    --batch-group-size "${NUM_BATCH_GROUP_SIZE}" \
    --num-workloads "${NUM_WORKLOADS}" \
    --methods "${method}" \
    --hybrid-top-k 0 \
    --worker-startup-timeout-sec "${WORKER_STARTUP_TIMEOUT}" \
    --worker-spawn-stagger-sec "${WORKER_SPAWN_STAGGER}"
}

run_text_method() {
  local method="$1"
  local exp_id="exp5b_text_dist_50agent_${method}"
  "${PYTHON_BIN}" decentralized_truth_discovery/examples/agent_scalability_distributed.py \
    --mode dist-tcp \
    --coord-addr "tcp://127.0.0.1:${TEXT_PORT}" \
    --input-json "${NUM_JSON}" \
    --text-mode \
    --input-text-json "${TEXT_JSON}" \
    --use-sbert \
    --output-root "${OUTPUT_ROOT}" \
    --experiment-id "${exp_id}" \
    --agent-counts "${AGENT_COUNTS}" \
    --subset-samples "${TEXT_SUBSET_SAMPLES}" \
    --batch-group-size "${TEXT_BATCH_GROUP_SIZE}" \
    --num-workloads "${TEXT_WORKLOADS}" \
    --methods "${method}" \
    --hybrid-top-k 0 \
    --worker-startup-timeout-sec "${WORKER_STARTUP_TIMEOUT}" \
    --worker-spawn-stagger-sec "${WORKER_SPAWN_STAGGER}"
}

merge_domain_outputs() {
  local domain="$1"
  local final_dir="$2"
  shift 2
  local methods=("$@")

  mkdir -p "${final_dir}"

  "${PYTHON_BIN}" - "$final_dir" "${methods[@]}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

final_dir = Path(sys.argv[1])
methods = sys.argv[2:]
domain = "numeric" if "numeric" in final_dir.name else "text"
frames = []
time_frames = []
for method in methods:
    method_dir = final_dir.parent / f"exp5b_{domain}_dist_50agent_{method}"
    raw_path = method_dir / "raw_runs.csv"
    time_path = method_dir / "summary_time_memory.csv"
    if not raw_path.exists():
        raise SystemExit(f"missing raw runs: {raw_path}")
    frames.append(pd.read_csv(raw_path))
    if time_path.exists():
        time_frames.append(pd.read_csv(time_path))

raw_df = pd.concat(frames, ignore_index=True)
raw_df.to_csv(final_dir / "raw_runs.csv", index=False)
if time_frames:
    pd.concat(time_frames, ignore_index=True).to_csv(
        final_dir / "summary_time_memory.csv",
        index=False,
    )
PY
}

NUMERIC_METHODS=(SenFeedTruth DecentTruth SenteTruth BasicTruth LASOTruth)
TEXT_METHODS=(BasicTruth LASOTruth SenteTruth)

NUMERIC_FINAL_DIR="${OUTPUT_ROOT}/exp5b_numeric_dist_50agent_time_memory"
TEXT_FINAL_DIR="${OUTPUT_ROOT}/exp5b_text_dist_50agent_time_memory"

case "${RUN_DOMAIN}" in
  both|numeric)
    for method in "${NUMERIC_METHODS[@]}"; do
      run_numeric_method "${method}"
    done
    merge_domain_outputs numeric "${NUMERIC_FINAL_DIR}" "${NUMERIC_METHODS[@]}"
    ;;
esac

case "${RUN_DOMAIN}" in
  both|text)
    for method in "${TEXT_METHODS[@]}"; do
      run_text_method "${method}"
    done
    merge_domain_outputs text "${TEXT_FINAL_DIR}" "${TEXT_METHODS[@]}"
    ;;
esac

if [[ "${RUN_DOMAIN}" != "both" && "${RUN_DOMAIN}" != "numeric" && "${RUN_DOMAIN}" != "text" ]]; then
  echo "Invalid RUN_DOMAIN=${RUN_DOMAIN}; expected both, numeric, or text." >&2
  exit 2
fi

echo "Done."
echo "Numeric output: ${NUMERIC_FINAL_DIR}"
echo "Text output   : ${TEXT_FINAL_DIR}"
