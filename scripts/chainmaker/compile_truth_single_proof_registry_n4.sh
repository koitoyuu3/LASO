#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTRACT_PATH="${CONTRACT_PATH:-$ROOT_DIR/decentralized_truth_discovery/core/circuits/TruthSingleProofRegistryN4.sol}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/decentralized_truth_discovery/core/circuits/build}"
SOLC_CMD="${SOLC_CMD:-}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$SOLC_CMD" ]]; then
  if command -v solc >/dev/null 2>&1; then
    SOLC_CMD="solc"
  elif command -v solcjs >/dev/null 2>&1; then
    SOLC_CMD="solcjs"
  else
    SOLC_CMD="npx --yes solc"
  fi
fi

echo "[compile] contract: $CONTRACT_PATH"
echo "[compile] output:   $OUTPUT_DIR"
echo "[compile] solc:     $SOLC_CMD"

if [[ "$SOLC_CMD" == "solc" ]]; then
  solc --evm-version london --abi --bin --overwrite -o "$OUTPUT_DIR" "$CONTRACT_PATH"
elif [[ "$SOLC_CMD" == "solcjs" ]]; then
  solcjs --evm-version london --abi --bin -o "$OUTPUT_DIR" "$CONTRACT_PATH"
else
  bash -lc "$SOLC_CMD --evm-version london --abi --bin --output-dir '$OUTPUT_DIR' '$CONTRACT_PATH'"
fi

echo "[compile] done"
