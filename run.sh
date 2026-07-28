#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${SHE_LORA_ENV:-she-lora}"
CONFIG_PATH="${PROJECT_ROOT}/config.yaml"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-127.0.0.1}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"

usage() {
    cat <<'EOF'
Usage: ./run.sh [command]

Commands:
  setup        Create the minimal Conda environment for the CKKS benchmark.
  keys         Generate the CKKS context and keys.
  benchmark    Run the local safetensors CKKS benchmark (default).
  tensorboard  Start TensorBoard for the benchmark logs.
  all          Run the benchmark, then start TensorBoard.
  help         Show this help message.

Before measuring 25% encryption, set this value in config.yaml:

  llm:
    ckks_benchmark:
      encryption_ratios: [0.25]

For a quick smoke test, also set max_clients and max_tensors to small positive
values. Leave both as null for the complete experiment.

Examples:
  ./run.sh setup
  ./run.sh benchmark
  ./run.sh tensorboard

Environment overrides:
  SHE_LORA_ENV=other-env ./run.sh benchmark
  TENSORBOARD_PORT=6007 ./run.sh tensorboard
EOF
}

require_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: conda is not available in PATH." >&2
        exit 1
    fi
}

require_environment() {
    require_conda
    if ! conda run -n "${ENV_NAME}" python -c "import sys" >/dev/null 2>&1; then
        echo "Error: Conda environment '${ENV_NAME}' is unavailable." >&2
        echo "Create it first with: ./run.sh setup" >&2
        exit 1
    fi
}

setup_environment() {
    require_conda
    if ! conda run -n "${ENV_NAME}" python -c "import sys" >/dev/null 2>&1; then
        conda create -n "${ENV_NAME}" python=3.11 pip -y
    fi
    conda run --no-capture-output -n "${ENV_NAME}" python -m pip install \
        "setuptools==80.9.0" \
        "numpy==2.2.4" \
        "torch==2.4.1" \
        "safetensors==0.5.3" \
        "tenseal==0.3.16" \
        "omegaconf==2.3.0" \
        "tensorboard==2.19.0"
}

generate_keys() {
    require_environment
    cd "${PROJECT_ROOT}"
    conda run --no-capture-output -n "${ENV_NAME}" \
        python -m flowertune_llm.she.gen_ckks_keys
}

run_benchmark() {
    require_environment
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "Error: project configuration not found: ${CONFIG_PATH}" >&2
        exit 1
    fi
    context_path="$({
        cd "${PROJECT_ROOT}"
        conda run -n "${ENV_NAME}" python -c \
            "from project_config import load_ckks_benchmark_config; print(load_ckks_benchmark_config(validate_paths=False).full_context_path)"
    })"
    if [[ ! -f "${context_path}" ]]; then
        echo "CKKS context not found; generating it before the benchmark."
        generate_keys
    fi
    cd "${PROJECT_ROOT}"
    conda run --no-capture-output -n "${ENV_NAME}" \
        python -m flowertune_llm.she.safetensors_benchmark
}

start_tensorboard() {
    require_environment
    if [[ ! -f "${CONFIG_PATH}" ]]; then
        echo "Error: project configuration not found: ${CONFIG_PATH}" >&2
        exit 1
    fi
    if ! conda run -n "${ENV_NAME}" python -c \
        "import pkg_resources, tensorboard" >/dev/null 2>&1; then
        echo "Error: TensorBoard dependencies are incomplete in '${ENV_NAME}'." >&2
        echo "Repair them with: ./run.sh setup" >&2
        exit 1
    fi
    tensorboard_dir="$({
        cd "${PROJECT_ROOT}"
        conda run -n "${ENV_NAME}" python -c \
            "from project_config import load_ckks_benchmark_config; print(load_ckks_benchmark_config(validate_paths=False).tensorboard_dir)"
    })"
    mkdir -p "${tensorboard_dir}"
    echo "TensorBoard: http://${TENSORBOARD_HOST}:${TENSORBOARD_PORT}"
    conda run --no-capture-output -n "${ENV_NAME}" tensorboard \
        --logdir "${tensorboard_dir}" \
        --host "${TENSORBOARD_HOST}" \
        --port "${TENSORBOARD_PORT}"
}

command="${1:-benchmark}"
case "${command}" in
    setup)
        setup_environment
        ;;
    keys)
        generate_keys
        ;;
    benchmark)
        run_benchmark
        ;;
    tensorboard)
        start_tensorboard
        ;;
    all)
        run_benchmark
        start_tensorboard
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Error: unknown command '${command}'." >&2
        usage >&2
        exit 2
        ;;
esac
