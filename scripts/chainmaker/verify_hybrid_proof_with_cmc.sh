#!/usr/bin/env bash
set -euo pipefail

CMC="${CMC:-cmc}"
SDK_CONF_PATH="${SDK_CONF_PATH:?SDK_CONF_PATH is required}"
ABI_PATH="${ABI_PATH:?ABI_PATH is required}"
CONTRACT_NAME="${CONTRACT_NAME:?CONTRACT_NAME is required}"
PARAMS_FILE="${PARAMS_FILE:?PARAMS_FILE is required}"
METHOD="${METHOD:-verifySelectedProofFlat}"
SYNC_RESULT="${SYNC_RESULT:-true}"
MODE="${MODE:-get}"

cmd=(
  "$CMC" client contract user "$MODE"
  --contract-name="$CONTRACT_NAME"
  --method="$METHOD"
  --sdk-conf-path="$SDK_CONF_PATH"
  --params="$(cat "$PARAMS_FILE")"
  --abi-file-path="$ABI_PATH"
)

if [[ "$MODE" == "invoke" ]]; then
  cmd+=(--sync-result="$SYNC_RESULT")
fi

echo "[verify] ${cmd[*]}"
"${cmd[@]}"
