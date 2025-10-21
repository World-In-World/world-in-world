#!/usr/bin/env bash
# NOTE: When launching this script, you need to execute two commands manually:
# unset LD_LIBRARY_PATH     # if an error occurs, try to unset LD_LIBRARY_PATH
# source /data/jieneng/software/anaconda/bin/activate /data/jieneng/software/miniconda3/envs/openeqa || exit

# ------------------------------------------------------------------
# Usage:
#   ./run_nav.sh <solver_module> <worker_num> <vllm_hosts> <WM_host> <gd_sam2_host> [extra_args...]
#
# Example:
# CUDA_VISIBLE_DEVICES="0" \
# bash downstream/scripts/init_solvers.sh \
#      downstream.solver \
#      08.12_ARF_Genex_igen_480 \
#      4 \
#      "n08:8000,n08:8001,n08:8002" \
#      n02:6000 \
#      n08:6002 \
#      --sam2_host=c006:6001 --use_WM &
# ------------------------------------------------------------------
set -euo pipefail

# --------------------------- 0. Check & assign ----------------------
if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <solver_module> <exp_id> <worker_num> <vllm_hosts> <WM_host> [extra_args...]" >&2
  exit 1
fi

solver_module="$1"
exp_id="$2"
worker_num="$3"
vllm_host_arg="$4"
WM_host="$5"
shift 5
extra_args=("$@")               # everything else

# Optional: convert comma list → Bash array if you ever need it
IFS=',' read -ra vllm_hosts_arr <<< "$vllm_host_arg"

# --------------------------- 1. Environment -------------------------
source downstream/scripts/set_env_variable.sh

# Try to initialise the module system and load gcc; continue on failure.
{
  source /apps/Lmod
  module load gcc/12.3.0
} || echo "Warning: could not source /apps/Lmod or load gcc/12.3.0; continuing without module environment." >&2

export PYTHONPATH="."

# --------------------------- 2. Logging -----------------------------
mkdir -p "downstream/logs/${exp_id}"

# --------------------------- 3. Launch ------------------------------
python -u -m "$solver_module" \
    --exp_id "$exp_id" \
    --worker_num "$worker_num" \
    --vllm_host  "${vllm_hosts_arr[@]}" \
    --WM_host "$WM_host" \
    --answerer_model "Qwen/Qwen2.5-VL-72B-Instruct-AWQ" \
    --planner_model  "Qwen/Qwen2.5-VL-72B-Instruct-AWQ" \
    --eval_model     "Qwen/Qwen2.5-VL-72B-Instruct-AWQ" \
    "${extra_args[@]}"

# OpenGVLab/InternVL3-78B-AWQ Qwen/Qwen2.5-VL-72B-Instruct-AWQ
# --------------------------- 4. GPU status --------------------------
nvidia-smi
