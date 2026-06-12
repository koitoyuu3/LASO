#!/bin/bash

set -euo pipefail

SSH_TARGET="${SSH_TARGET:-***@***.***.***.***}"
OLLAMA_BIN="${OLLAMA_BIN:-/home/***/ollama_offline/dist/bin/ollama}"
HOST_BIND="${HOST_BIND:-127.0.0.1}"
PORT_START="${PORT_START:-11431}"
BACKEND_COUNT="${BACKEND_COUNT:-4}"
GPU_COUNT="${GPU_COUNT:-4}"
OLLAMA_MODELS_DIR="${OLLAMA_MODELS_DIR:-/home/***/ollama_models/node5}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

ssh "${SSH_TARGET}" bash -s -- \
  "${OLLAMA_BIN}" \
  "${HOST_BIND}" \
  "${PORT_START}" \
  "${BACKEND_COUNT}" \
  "${GPU_COUNT}" \
  "${OLLAMA_MODELS_DIR}" \
  "${OLLAMA_NUM_PARALLEL}" \
  "${OLLAMA_KEEP_ALIVE}" <<'REMOTE'
set -euo pipefail

OLLAMA_BIN="$1"
HOST_BIND="$2"
PORT_START="$3"
BACKEND_COUNT="$4"
GPU_COUNT="$5"
OLLAMA_MODELS_DIR="$6"
OLLAMA_NUM_PARALLEL="$7"
OLLAMA_KEEP_ALIVE="$8"

if [ ! -x "${OLLAMA_BIN}" ]; then
  echo "ollama binary not executable: ${OLLAMA_BIN}" >&2
  exit 1
fi

if [ ! -d "${OLLAMA_MODELS_DIR}" ]; then
  echo "ollama models directory not found: ${OLLAMA_MODELS_DIR}" >&2
  exit 1
fi

get_pid_by_port() {
  local port="$1"
  ss -ltnp 2>/dev/null | awk -v port=":${port}" '$4 ~ port {print $NF}' \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1
}

wait_until_ready() {
  local port="$1"
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 3 "http://${HOST_BIND}:${port}/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

restart_backend() {
  local port="$1"
  local gpu="$2"
  local pid
  local log_file="/tmp/ollama_gpu_pool_${port}.log"

  pid="$(get_pid_by_port "${port}")"
  if [ -n "${pid}" ]; then
    kill "${pid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 10); do
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi

  nohup env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OLLAMA_HOST="${HOST_BIND}:${port}" \
    OLLAMA_MODELS="${OLLAMA_MODELS_DIR}" \
    OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL}" \
    OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE}" \
    "${OLLAMA_BIN}" serve >"${log_file}" 2>&1 </dev/null &

  echo "PORT ${port}: started on GPU ${gpu}, log=${log_file}"
}

echo "starting GPU-aware Ollama backend pool"
echo "target=${HOST_BIND}:${PORT_START}-$((${PORT_START} + ${BACKEND_COUNT} - 1))"
echo "models=${OLLAMA_MODELS_DIR}"
echo "gpu_count=${GPU_COUNT}"

for offset in $(seq 0 "$((${BACKEND_COUNT} - 1))"); do
  port="$((${PORT_START} + ${offset}))"
  gpu="$((${offset} % ${GPU_COUNT}))"
  restart_backend "${port}" "${gpu}"
done

echo
echo "verifying backend pool..."

failed=0
for offset in $(seq 0 "$((${BACKEND_COUNT} - 1))"); do
  port="$((${PORT_START} + ${offset}))"
  gpu="$((${offset} % ${GPU_COUNT}))"
  if ! wait_until_ready "${port}"; then
    echo "PORT ${port}: NOT_READY" >&2
    failed=1
    continue
  fi
  pid="$(get_pid_by_port "${port}")"
  models="$(curl -fsS --max-time 5 "http://${HOST_BIND}:${port}/api/tags" | python3 -c 'import sys,json; data=json.load(sys.stdin); print(", ".join(m.get("name", "") for m in data.get("models", [])) or "NO_MODELS")')"
  cuda_env=""
  if [ -n "${pid}" ]; then
    cuda_env="$(tr '\0' '\n' </proc/${pid}/environ | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n1)"
  fi
  echo "PORT ${port}: GPU=${gpu} ENV_GPU=${cuda_env:-unset} PID=${pid:-none} MODELS=${models}"
done

exit "${failed}"
REMOTE
