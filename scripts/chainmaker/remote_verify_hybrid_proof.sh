#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-***@***.***.***.***}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/tmp/truthfinder-chainmaker}"
REMOTE_CMC="${REMOTE_CMC:-cmc}"
REMOTE_SDK_CONF_PATH="${REMOTE_SDK_CONF_PATH:?REMOTE_SDK_CONF_PATH is required}"
REMOTE_CONTRACT_NAME="${REMOTE_CONTRACT_NAME:?REMOTE_CONTRACT_NAME is required}"
LOCAL_ABI_PATH="${LOCAL_ABI_PATH:?LOCAL_ABI_PATH is required}"
LOCAL_PARAMS_FILE="${LOCAL_PARAMS_FILE:?LOCAL_PARAMS_FILE is required}"
METHOD="${METHOD:-verifySelectedProofFlat}"
MODE="${MODE:-get}"

REMOTE_ABI_PATH="$REMOTE_WORKDIR/$(basename "$LOCAL_ABI_PATH")"
REMOTE_PARAMS_PATH="$REMOTE_WORKDIR/$(basename "$LOCAL_PARAMS_FILE")"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_WORKDIR'"
scp "$LOCAL_ABI_PATH" "$REMOTE_HOST:$REMOTE_ABI_PATH"
scp "$LOCAL_PARAMS_FILE" "$REMOTE_HOST:$REMOTE_PARAMS_PATH"

read -r -d '' REMOTE_SCRIPT <<EOS || true
set -euo pipefail
PARAMS=\$(cat '$REMOTE_PARAMS_PATH')
if [[ '$MODE' == 'invoke' ]]; then
  '$REMOTE_CMC' client contract user '$MODE' \
    --contract-name='$REMOTE_CONTRACT_NAME' \
    --method='$METHOD' \
    --sdk-conf-path='$REMOTE_SDK_CONF_PATH' \
    --params="\$PARAMS" \
    --abi-file-path='$REMOTE_ABI_PATH' \
    --sync-result=true
else
  '$REMOTE_CMC' client contract user '$MODE' \
    --contract-name='$REMOTE_CONTRACT_NAME' \
    --method='$METHOD' \
    --sdk-conf-path='$REMOTE_SDK_CONF_PATH' \
    --params="\$PARAMS" \
    --abi-file-path='$REMOTE_ABI_PATH'
fi
EOS

ssh "$REMOTE_HOST" "bash -lc $(printf '%q' "$REMOTE_SCRIPT")"
