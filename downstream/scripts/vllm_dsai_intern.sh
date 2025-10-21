#!/usr/bin/env bash
# Launch InternVL3-78B-AWQ with vLLM in OpenAI-compatible mode
# ------------------------------------------------------------
set -euo pipefail

# ---------- 0. BASIC ENVIRONMENT -------------------------------------------------
source downstream/scripts/set_env_variable.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True      # smoother large-ctx prefill
# add VLLM_ATTENTION_BACKEND=FLASHINFER
# export VLLM_ATTENTION_BACKEND=FLASHINFER  # use FlashInfer for faster attention

# ---------- 1. GPU / TENSOR PARALLEL SET-UP --------------------------------------
# If CUDA_VISIBLE_DEVICES is not preset, fall back to "all local GPUs"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | \
                                tr '\n' ',' | sed 's/,$//')
fi

# Derive tensor-parallel size from CUDA_VISIBLE_DEVICES
TP_NUM=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Using $TP_NUM GPU(s): $CUDA_VISIBLE_DEVICES"

# ---------- 2. MODEL & CONTEXT LENGTH -------------------------------------------
MODEL="OpenGVLab/InternVL3-78B-AWQ"
CONTEXT_LEN=22384

# ---------- 3. ENGINE ARGUMENT GROUPS -------------------------------------------
#  --kv-cache-dtype fp8 \
COMMON_ARGS="--model $MODEL \
             --allowed-local-media-path / \
             --trust-remote-code \
             --tensor-parallel-size $TP_NUM \
             --gpu-memory-utilization 0.94 \
             --max-model-len $CONTEXT_LEN \
             --max-num-batched-tokens $CONTEXT_LEN \
             --max-num-seqs 12 \
             --enforce-eager \
             --block-size 16 \
             --disable-custom-all-reduce"

# Add MM-specific cap (48 image tokens ≈ 3-4 full-HD images after V2PE)
MM_ARGS="--limit-mm-per-prompt image=48,video=12"

# Add quantization kernel only if the string “AWQ” is present
if [[ "$MODEL" == *"AWQ"* ]]; then
  COMMON_ARGS+=" --quantization awq"
fi

# ---------- 4. NETWORK -----------------------------------------------------------
PORT="${1:-8000}"          # default to 8000 if no CLI arg
COMMON_ARGS+=" --port $PORT"

# ---------- 5. RUN ---------------------------------------------------------------
LOG_DIR="downstream/logs/others"
mkdir -p "$LOG_DIR"

python3 -m vllm.entrypoints.openai.api_server \
        $COMMON_ARGS $MM_ARGS 2>&1 | tee "$LOG_DIR/vllm_${PORT}.log"

PID=$!
echo "==> vLLM server running on port ${PORT} (PID: ${PID})"
