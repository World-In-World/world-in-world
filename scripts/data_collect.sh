#!/bin/bash
nvidia-smi

# Run the Python script as a module
# --use_presaved_ep -> use presaved episodes. Note we use previous task episodes to get a better start position for data collection.
# --enable_depth -> collect depth data if set
# --trajs_num_per_scene -> number of trajs per scene (one loop across the whole scene is one traj, normally set to 1)

if [ "$#" -lt 3 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: bash scripts/data_collect.sh <exp_id> <output_dir> <num_processes>"
    echo "Example: bash scripts/data_collect.sh 09.30_debug data/datasets__/09.30_debug 8"
    exit 1
fi

EXP_ID="$1"
OUTPUT_DIR="$2"
NUM_PROCESSES="$3"

# Defaults (can be overridden via env)
DATASET_NAME="${DATASET_NAME:-hm3d-val}"
TRAJS_PER_SCENE="${TRAJS_PER_SCENE:-1}"
ENABLE_DEPTH_FLAG="${ENABLE_DEPTH_FLAG:---enable_depth}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUTPUT_DIR}"

run_job() {
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python -m habitat_data.HabitatRender \
        --exp_id "${EXP_ID}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_processes "${NUM_PROCESSES}" \
        --dataset_name "${DATASET_NAME}" \
        ${ENABLE_DEPTH_FLAG} \
        --trajs_num_per_scene "${TRAJS_PER_SCENE}"
}

echo "======>>> Running job."
curr_pid=$$
echo "======>>> The PID of this script is: <$curr_pid>"

run_job

echo "======>>> All process IDs in process group <${curr_pid}> (pid,ppid,pgid,cmd) are:"
ps -eo pid,ppid,pgid,cmd | awk -v pgid="$curr_pid" '$3 == pgid'
echo "======>>> End of process group listing."

sleep 7
nvidia-smi
echo "All jobs completed."
