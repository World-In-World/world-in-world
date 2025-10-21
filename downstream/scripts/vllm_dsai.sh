source downstream/scripts/set_env_variable.sh

# set tp_num compatible with CUDA_VISIBLE_DEVICES by code
export tp_num=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_USE_V1=0

port="$1"

model="Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
context_len=16384
context_args="--max-model-len $context_len --max-num-batched-tokens $context_len --port $port --generation-config vllm"
mm_args="--limit-mm-per-prompt image=48"
save_vram_args="--enable-chunked-prefill --enforce-eager"
fix_args="$model --allowed-local-media-path / --trust-remote-code -tp $tp_num --gpu-memory-utilization 0.95 --max-num-seqs 12"
if [[ $model == *"AWQ"* ]]; then
  fix_args="$fix_args -q awq_marlin"
fi
all_args="$fix_args $context_args $mm_args --disable-custom-all-reduce --block-size 16"

LOG_DIR="downstream/logs/others"
mkdir -p "$LOG_DIR"

python3 -m vllm.entrypoints.openai.api_server --model $all_args 2>&1 | tee $LOG_DIR/vllm_${port}.log

# Capture the PID of the last background process
pid=$!
echo "==> Job started with PID: ${pid}"

