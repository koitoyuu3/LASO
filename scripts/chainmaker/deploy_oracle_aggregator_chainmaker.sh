#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="${ROOT_DIR}/Backend"

CMC_BIN="${CMC_BIN:-${ROOT_DIR}/chainmaker-go/tools/cmc/cmc}"
SDK_CONF_PATH="${SDK_CONF_PATH:-src/main/resources/sdk_config.yml}"
CONTRACT_DIR="${CONTRACT_DIR:-${BACKEND_DIR}/src/main/resources/contract}"
ABI_FILE_PATH="${ABI_FILE_PATH:-${CONTRACT_DIR}/OracleAggregator.abi}"
BIN_FILE_PATH="${BIN_FILE_PATH:-${CONTRACT_DIR}/OracleAggregator.bin}"

CONTRACT_NAME="${CONTRACT_NAME:-OracleAggregator}"
CONTRACT_VERSION="${CONTRACT_VERSION:-}"
CHAIN_ID="${CHAIN_ID:-chain01}"
ORG_ID="${ORG_ID:-TestCMorg1}"

USER_TLS_CRT="${USER_TLS_CRT:-src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.tls.crt}"
USER_TLS_KEY="${USER_TLS_KEY:-src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.tls.key}"
USER_SIGN_CRT="${USER_SIGN_CRT:-src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.sign.crt}"
USER_SIGN_KEY="${USER_SIGN_KEY:-src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.sign.key}"
DEFAULT_ADMIN_KEY_FILE_PATHS="src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.tls.key,src/main/resources/crypto-config/TestCMorg2/user/cmtestuser2/cmtestuser2.tls.key,src/main/resources/crypto-config/TestCMorg3/user/cmtestuser3/cmtestuser3.tls.key"
DEFAULT_ADMIN_CRT_FILE_PATHS="src/main/resources/crypto-config/TestCMorg1/user/cmtestuser1/cmtestuser1.tls.crt,src/main/resources/crypto-config/TestCMorg2/user/cmtestuser2/cmtestuser2.tls.crt,src/main/resources/crypto-config/TestCMorg3/user/cmtestuser3/cmtestuser3.tls.crt"
DEFAULT_ADMIN_ORG_IDS="TestCMorg1,TestCMorg2,TestCMorg3"
ADMIN_KEY_FILE_PATHS="${ADMIN_KEY_FILE_PATHS:-${DEFAULT_ADMIN_KEY_FILE_PATHS}}"
ADMIN_CRT_FILE_PATHS="${ADMIN_CRT_FILE_PATHS:-${DEFAULT_ADMIN_CRT_FILE_PATHS}}"
ADMIN_ORG_IDS="${ADMIN_ORG_IDS:-${DEFAULT_ADMIN_ORG_IDS}}"

if [[ ! -x "${CMC_BIN}" ]]; then
  echo "cmc not found: ${CMC_BIN}" >&2
  exit 1
fi

if [[ ! -f "${ABI_FILE_PATH}" || ! -f "${BIN_FILE_PATH}" ]]; then
  echo "ABI/BIN missing, building artifacts first..."
  "${ROOT_DIR}/scripts/chainmaker/build_oracle_aggregator_artifacts.sh"
fi

increment_version() {
  local current="$1"

  if [[ "$current" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}.$((BASH_REMATCH[3] + 1))"
    return
  fi

  if [[ "$current" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}.$((BASH_REMATCH[2] + 1))"
    return
  fi

  if [[ "$current" =~ ^([0-9]+)$ ]]; then
    echo "$((BASH_REMATCH[1] + 1))"
    return
  fi

  echo "${current}.1"
}

cd "${BACKEND_DIR}"

set +e
contract_info="$(${CMC_BIN} query contract info "${CONTRACT_NAME}" --sdk-conf-path="${SDK_CONF_PATH}" 2>/dev/null)"
query_status=$?
set -e

action="create"
if [[ ${query_status} -eq 0 && -n "${contract_info}" ]]; then
  action="upgrade"
  current_version="$(printf '%s' "${contract_info}" | sed -n 's/.*"version": "\([^"]*\)".*/\1/p')"
  if [[ -z "${CONTRACT_VERSION}" ]]; then
    CONTRACT_VERSION="$(increment_version "${current_version}")"
  fi
else
  if [[ -z "${CONTRACT_VERSION}" ]]; then
    CONTRACT_VERSION="1.0.0"
  fi
fi

cmd=(
  "${CMC_BIN}" client contract user "${action}"
  "--contract-name=${CONTRACT_NAME}"
  "--runtime-type=EVM"
  "--byte-code-path=${BIN_FILE_PATH}"
  "--abi-file-path=${ABI_FILE_PATH}"
  "--version=${CONTRACT_VERSION}"
  "--sdk-conf-path=${SDK_CONF_PATH}"
  "--chain-id=${CHAIN_ID}"
  "--org-id=${ORG_ID}"
  "--user-tlscrt-file-path=${USER_TLS_CRT}"
  "--user-tlskey-file-path=${USER_TLS_KEY}"
  "--user-signcrt-file-path=${USER_SIGN_CRT}"
  "--user-signkey-file-path=${USER_SIGN_KEY}"
  "--sync-result=true"
)

if [[ "${action}" == "upgrade" && -n "${ADMIN_KEY_FILE_PATHS}" ]]; then
  cmd+=("--admin-key-file-paths=${ADMIN_KEY_FILE_PATHS}")
fi

if [[ "${action}" == "upgrade" && -n "${ADMIN_CRT_FILE_PATHS}" ]]; then
  cmd+=("--admin-crt-file-paths=${ADMIN_CRT_FILE_PATHS}")
fi

if [[ "${action}" == "upgrade" && -n "${ADMIN_ORG_IDS}" ]]; then
  cmd+=("--admin-org-ids=${ADMIN_ORG_IDS}")
fi

action_label="$(printf '%s' "${action}" | tr '[:lower:]' '[:upper:]')"
echo "${action_label} ${CONTRACT_NAME} with cmc, version=${CONTRACT_VERSION}..."
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
