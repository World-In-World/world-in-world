#!/usr/bin/env bash
# ------------------------------------------------------------
# Launch N parallel EmbodiedBench jobs, round-robin over VLLM
# back-ends and capture per-worker logs.
#
# Usage:
# source .bashrc || echo "Warning: .bashrc not found"
# ------------------------------------------------------------
set -euo pipefail

# --------------------------- 0. Check & assign ----------------------
if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <solver_env> <exp_id> <model_name> <num_workers> <vllm_hosts> <igenex_host> [extra_args...]" >&2
  exit 1
fi

solver_env="$1"
exp_id="$2"
model_name="$3"
num_workers="$4"
vllm_hosts_csv="$5"
igenex_host_csv="$6"
shift 6
extra_args=("$@")         # any additional Hydra/CLI args

# Set model_type to "remote" as default
model_type="remote"

# Turn CSV → Bash array
IFS=',' read -ra vllm_hosts <<< "$vllm_hosts_csv"
num_vllm_hosts="${#vllm_hosts[@]}"
IFS=',' read -ra igenex_hosts <<< "$igenex_host_csv"
num_igenex_hosts="${#igenex_hosts[@]}"

# Logs
LOG_ROOT="logs/${exp_id}"
mkdir -p "${LOG_ROOT}"/{out,err}

# --------------------------- helper -------------------------------
run_job() {
  # xvfb-run -a -s "-screen 0 1024x768x24" \
    python -m wiw_manip.main \
      --config-name="${solver_env}" \
      exp_name="${exp_id}" \
      model_name="${model_name}" \
      model_type="${model_type}" \
      igenex_host="${igenex_host}" \
      "${extra_args[@]}"
}

# --------------------------- main loop -----------------------------
for (( wid=0; wid<num_workers; wid++ )); do
  # 1. Round-robin VLLM assignment
  vidx=$(( wid % num_vllm_hosts ))
  vllm_host="${vllm_hosts[$vidx]}"
  remote_url_val="http://${vllm_host}/v1"

  # # 2. Round-robin igenex assignment
  iidx=$(( wid % num_igenex_hosts ))
  igenex_host="${igenex_hosts[$iidx]}"

  timestamp=$(date +"%m%d_%H%M%S")
  echo "==> Worker ${wid}/${num_workers} → vllm_url: ${remote_url_val} && igenex_url: ${igenex_host}"
  echo "==> Logging to ${LOG_ROOT}/out/out_${wid}-${timestamp}.log"

  (
    # Export env var **inside** subshell so it’s isolated
    export remote_url="${remote_url_val}"
    export DISPLAY=":99"
    run_job
  ) > >(tee    -a "${LOG_ROOT}/out/out_${wid}_${timestamp}.log") \
    2> >(tee -a "${LOG_ROOT}/err/err_${wid}_${timestamp}.log" >&2) &

  sleep 5s
done

wait
echo "All ${num_workers} jobs completed."
# sleep 5d
