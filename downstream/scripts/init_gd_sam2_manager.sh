#!/bin/bash
# Usage: bash downstream/scripts/init_managers.sh "4,5,6,7" num_workers_gd_sam2 exp_id
# where "4,5,6,7" is the CUDA_VISIBLE_DEVICES value,
# num_workers_gd_sam2 is the number of workers for the 'sam2' command,
# and exp_id is the experiment ID.

# Check that exactly 3 arguments are provided.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 num_workers_gd_sam2 exp_id"
    exit 1
fi

# Set input variables.
exp_id="$1"
num_workers_gd_sam2="$2"

source downstream/scripts/set_env_variable.sh

# Automatically determine the host IP using the first available IP.
host="0.0.0.0"
# host="127.0.0.1"

# Function to find an available TCP port starting from 6000.
find_free_port() {
    local skip="$1"
    local port=6002
    while true; do
        # If the current port equals the port to skip, move to the next.
        if [ "$port" -eq "$skip" ]; then
            ((port++))
            continue
        fi
        # Check if the port is free using lsof.
        if ! lsof -i tcp:"$port" &> /dev/null; then
            echo "$port"
            break
        fi
        ((port++))
    done
}

# Set up a trap to kill all child processes if the script exits.
trap 'kill -- -$$' SIGINT SIGTERM EXIT

# Array to store PIDs of the jobs.
declare -a pids=()

# Find a free port for the sam2 command.
port=$(find_free_port 0)

# Echo the determined host and port.
echo "Using host and port for 'sam2': $host:$port"
echo "current machine Hostname -I: $(hostname -I):$port"

# Run the sam2 worker manager command in the background.
PYTHONPATH="." \
python downstream/utils/worker_manager.py \
    --host "$host" --port "$port" \
    --num_workers "$num_workers_gd_sam2" \
    --exp_id "$exp_id" \
    --worker_type gd_sam2

# Capture the PID of the last background process.
pid=$!
echo "Job started with PID: ${pid}"
pids+=("${pid}")

echo "Waiting for all jobs to complete..."
for pid in "${pids[@]}"; do
    wait "${pid}"
    echo "Job with PID ${pid} should have completed."
    echo "======>>> All process IDs in process group <${pid}> (pid,ppid,pgid,cmd) are:"
    ps -eo pid,ppid,pgid,cmd | awk -v pgid="$pid" '$3 == pgid'
    echo "======>>> End of process group listing."
done

nvidia-smi
echo "All jobs completed."

# Keep the script alive for a while (e.g., 5 days).
sleep 5d
