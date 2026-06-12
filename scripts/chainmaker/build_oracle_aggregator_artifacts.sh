#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTRACT_DIR="${ROOT_DIR}/Backend/src/main/resources/contract"
SOLC_NPM_PACKAGE="${SOLC_NPM_PACKAGE:-solc@0.8.13}"

cd "${CONTRACT_DIR}"

echo "Compiling OracleAggregator.sol into ABI/BIN with ${SOLC_NPM_PACKAGE}..."
npm_config_registry=https://registry.npmjs.org npx --yes "${SOLC_NPM_PACKAGE}" --bin --abi OracleAggregator.sol --output-dir .

cp -f OracleAggregator_sol_OracleAggregator.abi OracleAggregator.abi
cp -f OracleAggregator_sol_OracleAggregator.bin OracleAggregator.bin

echo "Artifacts generated:"
echo "- ${CONTRACT_DIR}/OracleAggregator.abi"
echo "- ${CONTRACT_DIR}/OracleAggregator.bin"
echo "- ${CONTRACT_DIR}/OracleAggregator_sol_OracleAggregator.abi"
echo "- ${CONTRACT_DIR}/OracleAggregator_sol_OracleAggregator.bin"
