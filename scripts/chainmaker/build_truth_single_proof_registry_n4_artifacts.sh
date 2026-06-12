#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTRACT_DIR="${ROOT_DIR}/Backend/src/main/resources/contract"
CONTRACT_SOURCE="${CONTRACT_SOURCE:-TruthSingleProofRegistryN4.sol}"
CONTRACT_BASENAME="${CONTRACT_BASENAME:-TruthSingleProofRegistryN4}"
SOLC_NPM_PACKAGE="${SOLC_NPM_PACKAGE:-solc@0.8.20}"
SOLC_BIN="${SOLC_BIN:-}"
EVM_VERSION="${EVM_VERSION:-london}"
INPUT_JSON_FILE="$(mktemp)"
OUTPUT_JSON_FILE="$(mktemp)"

cleanup() {
  rm -f "${INPUT_JSON_FILE}" "${OUTPUT_JSON_FILE}"
}

trap cleanup EXIT

cd "${CONTRACT_DIR}"

echo "Compiling ${CONTRACT_SOURCE} into ABI/BIN with ${SOLC_NPM_PACKAGE} (evm=${EVM_VERSION})..."
python3 - "${CONTRACT_SOURCE}" "${EVM_VERSION}" > "${INPUT_JSON_FILE}" <<'PY'
import json
import pathlib
import sys

source_name = sys.argv[1]
evm_version = sys.argv[2]
source_path = pathlib.Path(source_name)

print(json.dumps({
    "language": "Solidity",
    "sources": {
        source_name: {
            "content": source_path.read_text(encoding="utf-8"),
        }
    },
    "settings": {
        "evmVersion": evm_version,
        "viaIR": True,
        "optimizer": {
            "enabled": True,
            "runs": 200,
        },
        "outputSelection": {
            "*": {
                "*": ["abi", "evm.bytecode.object"],
            }
        },
    },
}, ensure_ascii=False))
PY

if [[ -n "${SOLC_BIN}" ]]; then
  "${SOLC_BIN}" --standard-json < "${INPUT_JSON_FILE}" > "${OUTPUT_JSON_FILE}"
else
  npm_config_registry=https://registry.npmjs.org npx --yes "${SOLC_NPM_PACKAGE}" --standard-json < "${INPUT_JSON_FILE}" > "${OUTPUT_JSON_FILE}"
fi

python3 - "${OUTPUT_JSON_FILE}" "${CONTRACT_SOURCE}" "${CONTRACT_BASENAME}" <<'PY'
import json
import pathlib
import sys

output_path = pathlib.Path(sys.argv[1])
source_name = sys.argv[2]
contract_name = sys.argv[3]
source_prefix = pathlib.Path(source_name).stem

raw_output = output_path.read_text(encoding="utf-8")
json_start = raw_output.find("{")
if json_start < 0:
    raise SystemExit(f"compiler did not produce JSON output:\n{raw_output}")
data = json.loads(raw_output[json_start:])
errors = [item for item in data.get("errors", []) if item.get("severity") == "error"]
if errors:
    for item in errors:
        print(item.get("formattedMessage") or item.get("message") or json.dumps(item, ensure_ascii=False), file=sys.stderr)
    raise SystemExit(1)

contract = (((data.get("contracts") or {}).get(source_name) or {}).get(contract_name))
if not contract:
    raise SystemExit(f"contract not found in compiler output: {source_name}:{contract_name}")

abi_path = pathlib.Path(f"{source_prefix}_sol_{contract_name}.abi")
bin_path = pathlib.Path(f"{source_prefix}_sol_{contract_name}.bin")
short_abi_path = pathlib.Path(f"{contract_name}.abi")
short_bin_path = pathlib.Path(f"{contract_name}.bin")

abi_text = json.dumps(contract["abi"], ensure_ascii=False)
bin_text = contract["evm"]["bytecode"]["object"]

abi_path.write_text(abi_text, encoding="utf-8")
bin_path.write_text(bin_text, encoding="utf-8")
short_abi_path.write_text(abi_text, encoding="utf-8")
short_bin_path.write_text(bin_text, encoding="utf-8")
PY

echo "Artifacts generated:"
echo "- ${CONTRACT_DIR}/${CONTRACT_BASENAME}.abi"
echo "- ${CONTRACT_DIR}/${CONTRACT_BASENAME}.bin"
echo "- ${CONTRACT_DIR}/${CONTRACT_BASENAME}_sol_${CONTRACT_BASENAME}.abi"
echo "- ${CONTRACT_DIR}/${CONTRACT_BASENAME}_sol_${CONTRACT_BASENAME}.bin"
