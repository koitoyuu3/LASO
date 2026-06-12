#!/usr/bin/env bash
set -euo pipefail

CMC="${CMC:-cmc}"
CONTRACT_NAME="${CONTRACT_NAME:-truth_single_proof_registry_n4}"
CONTRACT_VERSION="${CONTRACT_VERSION:-1.0}"
SYNC_RESULT="${SYNC_RESULT:-true}"
SDK_CONF_PATH="${SDK_CONF_PATH:?SDK_CONF_PATH is required}"
BIN_PATH="${BIN_PATH:?BIN_PATH is required}"
ABI_PATH="${ABI_PATH:?ABI_PATH is required}"
ADMIN_KEY_FILE_PATHS="${ADMIN_KEY_FILE_PATHS:-}"
ADMIN_CRT_FILE_PATHS="${ADMIN_CRT_FILE_PATHS:-}"

cmd=(
  "$CMC" client contract user create
  --contract-name="$CONTRACT_NAME"
  --runtime-type=EVM
  --byte-code-path="$BIN_PATH"
  --abi-file-path="$ABI_PATH"
  --version="$CONTRACT_VERSION"
  --sdk-conf-path="$SDK_CONF_PATH"
  --sync-result="$SYNC_RESULT"
)

if [[ -n "$ADMIN_KEY_FILE_PATHS" ]]; then
  cmd+=(--admin-key-file-paths="$ADMIN_KEY_FILE_PATHS")
fi
if [[ -n "$ADMIN_CRT_FILE_PATHS" ]]; then
  cmd+=(--admin-crt-file-paths="$ADMIN_CRT_FILE_PATHS")
fi

echo "[deploy] ${cmd[*]}"
"${cmd[@]}"
