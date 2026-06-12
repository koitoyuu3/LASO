#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-***@***.***.***.***}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/tmp/truthfinder-chainmaker}"
REMOTE_CMC="${REMOTE_CMC:-cmc}"
REMOTE_SDK_CONF_PATH="${REMOTE_SDK_CONF_PATH:?REMOTE_SDK_CONF_PATH is required}"
REMOTE_ADMIN_KEY_FILE_PATHS="${REMOTE_ADMIN_KEY_FILE_PATHS:-}"
REMOTE_ADMIN_CRT_FILE_PATHS="${REMOTE_ADMIN_CRT_FILE_PATHS:-}"
CONTRACT_NAME="${CONTRACT_NAME:-truth_single_proof_registry_n4}"
CONTRACT_VERSION="${CONTRACT_VERSION:-1.0}"
LOCAL_BIN_PATH="${LOCAL_BIN_PATH:?LOCAL_BIN_PATH is required}"
LOCAL_ABI_PATH="${LOCAL_ABI_PATH:?LOCAL_ABI_PATH is required}"

REMOTE_BIN_PATH="$REMOTE_WORKDIR/$(basename "$LOCAL_BIN_PATH")"
REMOTE_ABI_PATH="$REMOTE_WORKDIR/$(basename "$LOCAL_ABI_PATH")"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_WORKDIR'"
scp "$LOCAL_BIN_PATH" "$REMOTE_HOST:$REMOTE_BIN_PATH"
scp "$LOCAL_ABI_PATH" "$REMOTE_HOST:$REMOTE_ABI_PATH"

remote_cmd=(
  "$REMOTE_CMC" client contract user create
  --contract-name="$CONTRACT_NAME"
  --runtime-type=EVM
  --byte-code-path="$REMOTE_BIN_PATH"
  --abi-file-path="$REMOTE_ABI_PATH"
  --version="$CONTRACT_VERSION"
  --sdk-conf-path="$REMOTE_SDK_CONF_PATH"
  --sync-result=true
)

if [[ -n "$REMOTE_ADMIN_KEY_FILE_PATHS" ]]; then
  remote_cmd+=(--admin-key-file-paths="$REMOTE_ADMIN_KEY_FILE_PATHS")
fi
if [[ -n "$REMOTE_ADMIN_CRT_FILE_PATHS" ]]; then
  remote_cmd+=(--admin-crt-file-paths="$REMOTE_ADMIN_CRT_FILE_PATHS")
fi

ssh "$REMOTE_HOST" "${remote_cmd[*]}"
