#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/Backend"

CMC_BIN="${CMC_BIN:-${ROOT_DIR}/chainmaker-go/tools/cmc/cmc}"
SDK_CONF_PATH="${SDK_CONF_PATH:-src/main/resources/sdk_config.yml}"
ABI_FILE_PATH="${ABI_FILE_PATH:-${ROOT_DIR}/Backend/src/main/resources/contract/TruthSingleProofRegistryN4.abi}"

CONTRACT_NAME="${CONTRACT_NAME:-TruthSingleProofRegistryN4}"
CHAIN_ID="${CHAIN_ID:-chain01}"
ORG_ID="${ORG_ID:-TestCMorg1}"
GROUP_INDEX="${GROUP_INDEX:-all}"
SUBMIT_ON_SUCCESS="${SUBMIT_ON_SUCCESS:-true}"
RESULT_URI_PREFIX="${RESULT_URI_PREFIX:-bundle://result/}"
PROOF_URI_PREFIX="${PROOF_URI_PREFIX:-bundle://proof/}"

USER_TLS_CRT="${USER_TLS_CRT:-src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.tls.crt}"
USER_TLS_KEY="${USER_TLS_KEY:-src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.tls.key}"
USER_SIGN_CRT="${USER_SIGN_CRT:-src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.sign.crt}"
USER_SIGN_KEY="${USER_SIGN_KEY:-src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.sign.key}"

PROOF_BUNDLE_PATH="${PROOF_BUNDLE_PATH:-${1:-}}"

if [[ -z "${PROOF_BUNDLE_PATH}" ]]; then
  echo "usage: PROOF_BUNDLE_PATH=/path/to/*.proof_bundle.json $0" >&2
  echo "   or: $0 /path/to/*.proof_bundle.json" >&2
  exit 1
fi

if [[ ! -f "${PROOF_BUNDLE_PATH}" ]]; then
  echo "proof bundle not found: ${PROOF_BUNDLE_PATH}" >&2
  exit 1
fi

if [[ ! -x "${CMC_BIN}" ]]; then
  echo "cmc not found: ${CMC_BIN}" >&2
  exit 1
fi

if [[ ! -f "${ABI_FILE_PATH}" ]]; then
  echo "ABI missing, building artifacts first..."
  "${ROOT_DIR}/scripts/chainmaker/build_truth_single_proof_registry_n4_artifacts.sh"
fi

cd "${BACKEND_DIR}"

extract_contract_result() {
  python3 -c 'import json,sys; data=json.loads(sys.stdin.read()); print((data.get("contract_result") or {}).get("result", ""))'
}

strip_array_wrapper() {
  local value="$1"
  value="${value#[}"
  value="${value%]}"
  printf '%s' "${value}"
}

invoke_contract() {
  local method="$1"
  local params="$2"

  "${CMC_BIN}" client contract user invoke \
    --contract-name="${CONTRACT_NAME}" \
    --method="${method}" \
    --sdk-conf-path="${SDK_CONF_PATH}" \
    --chain-id="${CHAIN_ID}" \
    --org-id="${ORG_ID}" \
    --user-tlscrt-file-path="${USER_TLS_CRT}" \
    --user-tlskey-file-path="${USER_TLS_KEY}" \
    --user-signcrt-file-path="${USER_SIGN_CRT}" \
    --user-signkey-file-path="${USER_SIGN_KEY}" \
    --sync-result=true \
    --abi-file-path="${ABI_FILE_PATH}" \
    --params="${params}"
}

get_contract() {
  local method="$1"
  local params="$2"

  "${CMC_BIN}" client contract user get \
    --contract-name="${CONTRACT_NAME}" \
    --method="${method}" \
    --sdk-conf-path="${SDK_CONF_PATH}" \
    --chain-id="${CHAIN_ID}" \
    --org-id="${ORG_ID}" \
    --user-tlscrt-file-path="${USER_TLS_CRT}" \
    --user-tlskey-file-path="${USER_TLS_KEY}" \
    --user-signcrt-file-path="${USER_SIGN_CRT}" \
    --user-signkey-file-path="${USER_SIGN_KEY}" \
    --abi-file-path="${ABI_FILE_PATH}" \
    --params="${params}"
}

calc_submission_key() {
  local experiment_id="$1"
  local group_id="$2"
  local proof_id="$3"
  local helper="/tmp/calc_submission_key.go"

  cat > "${helper}" <<'EOF'
package main

import (
    "encoding/hex"
    "fmt"
    "os"
    "strings"

    "golang.org/x/crypto/sha3"
)

func main() {
    if len(os.Args) != 4 {
        panic("usage: calc_submission_key <experiment_id> <group_id> <proof_id>")
    }

    hasher := sha3.NewLegacyKeccak256()
    for _, raw := range os.Args[1:] {
        value := strings.TrimPrefix(raw, "0x")
        decoded, err := hex.DecodeString(value)
        if err != nil {
            panic(err)
        }
        if len(decoded) != 32 {
            panic("bytes32 argument must be 32 bytes")
        }
        _, _ = hasher.Write(decoded)
    }

    fmt.Printf("0x%x", hasher.Sum(nil))
}
EOF

  (
    cd "${ROOT_DIR}/chainmaker-go"
    GOWORK=off GOCACHE=/tmp/go-build-cache go run "${helper}" "${experiment_id}" "${group_id}" "${proof_id}"
  )
}

response_ok() {
  python3 -c 'import json,sys; data=json.load(sys.stdin); code=data.get("code", 0); ccode=(data.get("contract_result") or {}).get("code", 0); sys.exit(0 if code in (0, None) and ccode in (0, None) else 1)'
}

build_group_rows() {
  python3 - "${PROOF_BUNDLE_PATH}" "${GROUP_INDEX}" "${RESULT_URI_PREFIX}" "${PROOF_URI_PREFIX}" <<'PY'
import base64
import json
import sys

path, group_index, result_uri_prefix, proof_uri_prefix = sys.argv[1:5]

with open(path, 'r', encoding='utf-8') as fh:
    bundle = json.load(fh)

groups = bundle.get('proof_groups') or []
if not groups:
    raise SystemExit('proof_groups is empty')

if group_index == 'all':
    indexes = range(len(groups))
else:
    indexes = [int(group_index)]

for idx in indexes:
    group = groups[idx]
    submission = group['contract_submission']
    scalars = [
        submission['a'][0], submission['a'][1],
        submission['b'][0][0], submission['b'][0][1],
        submission['b'][1][0], submission['b'][1][1],
        submission['c'][0], submission['c'][1],
        *submission['pubSignals'],
    ]
    verify_params = [{"uint256": str(value)} for value in scalars]
    submit_params = [
        {"bytes32": submission['experimentId']},
        {"bytes32": submission['groupId']},
        {"bytes32": submission['resultDigest']},
        {"bytes32": submission['proofId']},
        {"bytes32": submission['statementDigestSha256']},
        {"string": f"{result_uri_prefix}{path}#group={idx}"},
        {"string": f"{proof_uri_prefix}{path}#group={idx}"},
        *verify_params,
    ]
    key_params = [
        {"bytes32": submission['experimentId']},
        {"bytes32": submission['groupId']},
        {"bytes32": submission['proofId']},
    ]
    row = {
        'index': idx,
        'proof_id': group['proof_id'],
        'group_digest': group['group_digest'],
        'experiment_id': submission['experimentId'],
        'contract_group_id': submission['groupId'],
        'contract_proof_id': submission['proofId'],
        'verify_params': json.dumps(verify_params, ensure_ascii=False, separators=(',', ':')),
        'submit_params': json.dumps(submit_params, ensure_ascii=False, separators=(',', ':')),
        'key_params': json.dumps(key_params, ensure_ascii=False, separators=(',', ':')),
    }
    print(base64.b64encode(json.dumps(row, ensure_ascii=False).encode('utf-8')).decode('ascii'))
PY
}

json_field() {
  local encoded="$1"
  local field="$2"
  python3 - "${encoded}" "${field}" <<'PY'
import base64
import json
import sys

record = json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'))
print(record[sys.argv[2]])
PY
}

verified_count=0
submitted_count=0
skipped_count=0

while IFS= read -r row; do
  [[ -z "${row}" ]] && continue

  group_index_value="$(json_field "${row}" index)"
  proof_id="$(json_field "${row}" proof_id)"
  group_digest="$(json_field "${row}" group_digest)"
  experiment_id="$(json_field "${row}" experiment_id)"
  contract_group_id="$(json_field "${row}" contract_group_id)"
  contract_proof_id="$(json_field "${row}" contract_proof_id)"
  verify_params="$(json_field "${row}" verify_params)"
  submit_params="$(json_field "${row}" submit_params)"
  key_params="$(json_field "${row}" key_params)"

  echo "=== group ${group_index_value} (${group_digest}) proof ${proof_id} ==="

  verify_json="$(get_contract "verifySelectedProofFlat" "${verify_params}")"
  verify_result="$(strip_array_wrapper "$(printf '%s' "${verify_json}" | extract_contract_result)")"
  echo "verifySelectedProofFlat=${verify_result}"

  if [[ "${verify_result}" != "true" ]]; then
    echo "proof verification failed for group ${group_index_value}" >&2
    exit 1
  fi
  verified_count=$((verified_count + 1))

  submission_key="$(calc_submission_key "${experiment_id}" "${contract_group_id}" "${contract_proof_id}")"
  echo "submissionKey=${submission_key}"

  has_submission_json="$(get_contract "hasSubmission" "[{\"bytes32\":\"${submission_key}\"}]")"
  has_submission="$(strip_array_wrapper "$(printf '%s' "${has_submission_json}" | extract_contract_result)")"
  echo "hasSubmission(before)=${has_submission}"

  if [[ "${has_submission}" == "true" ]]; then
    echo "submission already exists, skip invoke"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  if [[ "${SUBMIT_ON_SUCCESS}" != "true" ]]; then
    echo "SUBMIT_ON_SUCCESS=${SUBMIT_ON_SUCCESS}, skip invoke"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  submit_json="$(invoke_contract "submitSelectedProofFlat" "${submit_params}")"
  printf 'submitSelectedProofFlat response=%s\n' "${submit_json}"
  if ! printf '%s' "${submit_json}" | response_ok; then
    echo "submitSelectedProofFlat failed for group ${group_index_value}" >&2
    exit 1
  fi

  has_submission_json="$(get_contract "hasSubmission" "[{\"bytes32\":\"${submission_key}\"}]")"
  has_submission="$(strip_array_wrapper "$(printf '%s' "${has_submission_json}" | extract_contract_result)")"
  echo "hasSubmission(after)=${has_submission}"
  if [[ "${has_submission}" != "true" ]]; then
    echo "submission not persisted for group ${group_index_value}" >&2
    exit 1
  fi

  submitted_count=$((submitted_count + 1))
done < <(build_group_rows)

echo "SUCCESS verified=${verified_count} submitted=${submitted_count} skipped=${skipped_count}"
