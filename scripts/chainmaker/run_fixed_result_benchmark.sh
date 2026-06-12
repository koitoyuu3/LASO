#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8080}"
RESULT_PATH="${RESULT_PATH:-${ROOT_DIR}/input-data/result_request15_llama31_8b.json}"
PROOF_BUNDLE_PATH="${PROOF_BUNDLE_PATH:-}"
PROMPT="${PROMPT:-benchmark fixed AI result request}"
CHAINS="${CHAINS:-chainmaker}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark-results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"

for chain in ${CHAINS}; do
  output_file="${OUTPUT_DIR}/${RUN_TAG}_${chain}_fixed_result.json"
  curl --silent --show-error --location --request POST \
    --get "${BACKEND_URL}/api/benchmark/ollama-fixed-result" \
    --data-urlencode "chain=${chain}" \
    --data-urlencode "resultPath=${RESULT_PATH}" \
    ${PROOF_BUNDLE_PATH:+--data-urlencode "proofBundlePath=${PROOF_BUNDLE_PATH}"} \
    --data-urlencode "prompt=${PROMPT}" \
    --output "${output_file}"
  echo "${chain}: ${output_file}"
done
