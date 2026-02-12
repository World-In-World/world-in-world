#!/usr/bin/env bash
# ------------------------------------------------------------------
# Launch a *worldmodel* manager.
#
# Usage:
#   CUDA_VISIBLE_DEVICES="0,1,2,3" \
#   bash downstream/scripts/init_worldmodel_manager.sh \
#       07.02_iGnav 4 ltx --task_type=manipulation &
#
# Args:
#   1: exp_id        (string, e.g., 07.02_iGnav)
#   2: num_workers   (int)
#   3: worker_type   (string: igenex|igenex_manip|se3ds|pathdreamer|nwm|hunyuan|ltx|wan21 ...) (search WORLD_MODEL_TYPES in this project all available options)
#   4+: extra args forwarded verbatim to worker_manager.py
#
# CUDA_VISIBLE_DEVICES is honored and forwarded implicitly via env.
# ------------------------------------------------------------------
source downstream/scripts/set_env_variable.sh

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <exp_id> <num_workers> <worker_type> [extra_args...]" >&2
    exit 1
fi

exp_id="$1"
num_workers_igenex="$2"
workers_type="$3"
shift 3
extra_args=("$@")        # may be empty

# Default host binding (0.0.0.0 for remote access; change to 127.0.0.1 if local only).
host="0.0.0.0"
# host="127.0.0.1"

# ------------------------------------------------------------------
# Find a free TCP port starting at 7000 (skipping an optional single port).
# Usage: find_free_port <skip_port_or_0>
# ------------------------------------------------------------------
find_free_port() {
    local skip="$1"
    local port=7000
    while true; do
        if [[ "$port" -eq "$skip" ]]; then
            ((port++))
            continue
        fi
        if ! lsof -i tcp:"$port" &>/dev/null; then
            echo "$port"
            return 0
        fi
        ((port++))
    done
}

# Set up a trap to kill all child processes if the script exits.
trap 'kill -- -$$ 2>/dev/null || true' SIGINT SIGTERM EXIT

declare -a pids=()

port="$(find_free_port 0)"

echo "Using host and port: ${host}:${port}"
echo "Hostname -I: $(hostname -I)  (manager port: $port)"
if (( ${#extra_args[@]} )); then
    echo "Forwarding extra args to manager: ${extra_args[*]}"
fi

# ------------------------------------------------------------------
# Launch manager (backgrounded so we can wait later).
# NOTE: we intentionally *do not* quote PYTHONPATH assignment so that '.'
# becomes the path prefix; adjust if your env differs.
# ------------------------------------------------------------------
PYTHONPATH="." \
python downstream/utils/worker_manager.py \
    --host "$host" \
    --port "$port" \
    --num_workers "$num_workers_igenex" \
    --exp_id "$exp_id" \
    --worker_type "$workers_type" \
    "${extra_args[@]}" &

pid=$!
echo "Manager started (PID ${pid})."
pids+=("$pid")

echo "Waiting for all jobs to complete..."
for pid in "${pids[@]}"; do
    if wait "$pid"; then
        echo "Job with PID ${pid} exited normally."
    else
        echo "Job with PID ${pid} exited with error ($?)."
    fi
    echo "======>>> Process group <${pid}> (pid,ppid,pgid,cmd):"
    ps -eo pid,ppid,pgid,cmd | awk -v pgid="$pid" '$3 == pgid'
    echo "======>>> End process group listing."
done

nvidia-smi
echo "All jobs completed."

# Keep the shell alive for monitoring (5 days); adjust as needed.
sleep 5d
