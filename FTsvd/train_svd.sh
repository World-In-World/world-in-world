#!/bin/bash

TAG="08.02_0.01D_uni_full_noFPSid_3acm_aug_NoiseA_1e-5"

nvidia-smi

optim_seeds_array=(1)

# Array to store PIDs of the  jobs
declare -a pids=()

# Create a function to run the job
run_job() {
    # Set environment variables inline, then run the command
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    # WANDB_API_KEY=XXXX \
    python -m accelerate.commands.launch --config_file="FTsvd/config/accelerate_deepspeed_o1_config.yaml" \
        FTsvd/train_svd.py \
        --exp_id "${TAG}" \
        --seed "${seed}" \
        --learning_rate=2e-5  \
        --width=1024 \
        --height=576 \
        --num_frames 14 \
        --mixed_precision="bf16" \
        --per_gpu_batch_size=1 --gradient_accumulation_steps=4 \
        --checkpointing_steps=1000 --validation_steps=500 --checkpoints_total_limit=1 \
        --num_workers=8 \
        --num_past_obs=1 \
        --train_param_type="full" \
        --action_strategy="micro_cond" \
        --action_input_channel=14 \
        --task_type="navigation" \
        --max_train_steps=1002 \
        --lr_scheduler=cosine \
        --base_folder <data/datasets/habitat_tasks/nav_dataset_train_split_new> \
        --val_base_folder <data/datasets/habitat_tasks/nav_dataset_train_split_new> \
        --report_to=wandb \
        --resume_from_checkpoint=${ENABLE_RESUME} \
        --output_dir=${OUTPUT_DIR} \
        --enable_reverse_aug \
        --pretrained_model_name_or_path \
            FTsvd/ckpts_upload/models--stabilityai--stable-video-diffusion-img2vid

    # Capture the PID of the last background process
    local pid=$!
    echo "Job for seed ${seed} started with PID: ${pid}"
    pids+=("${pid}")  # Store PID in the array
}

# Loop over each seed (or other parameters if needed)
for seed in "${optim_seeds_array[@]}"; do

    # Check whether TAG contains the substring "resume"
    if [[ "$TAG" == *"resume"* ]]; then
        OUTPUT_DIR="outputs/07.22_0.1D_uni_full_noFPSid_3acm_aug_NoiseA/seed_1_0722_213129"
        ENABLE_RESUME="latest"
    else
        # Example: define an output directory based on the seed
        OUTPUT_DIR="./outputs/${TAG}/seed_${seed}"
        ENABLE_RESUME=None
    fi
    echo "Output directory is: ${OUTPUT_DIR}"

    # If you want to skip jobs if the directory already exists:
    if [ -d "${OUTPUT_DIR}" ] && [ "${ENABLE_RESUME}" == "None" ]; then
        echo "------------"
        echo "Results found in ${OUTPUT_DIR}, adding time as postfix."
        OUTPUT_DIR="${OUTPUT_DIR}_$(date +%m%d_%H%M%S)"
        echo "New output directory is: ${OUTPUT_DIR}"
        echo "------------"
    fi

    echo "======>>> Running job with seed=${seed}. Output will go to ${OUTPUT_DIR}."
    mkdir -p "${OUTPUT_DIR}"
    # curr_pid=$$
    # echo "======>>> The PID of this script is: <$curr_pid>"
    JOB_ID=${SLURM_JOB_ID:-local}
    {
    run_job
    } > >(tee -a "${OUTPUT_DIR}/out_${JOB_ID}.log") 2> >(tee -a "${OUTPUT_DIR}/err_${JOB_ID}.log" >&2)

    # Optional: Wait for all background jobs to finish before exiting the script
    # Uncomment if you run jobs in parallel and want to wait for all to complete
    echo "Waiting for all jobs to complete..."
    for pid in "${pids[@]}"; do
        wait "${pid}"
        echo "Job with PID ${pid} should have completed."

        echo "======>>> All process IDs in process group <${pid}> (pid,ppid,pgid,cmd) are:"
        ps -eo pid,ppid,pgid,cmd | awk -v pgid="$pid" '$3 == pgid'
        echo "======>>> End of process group listing."
    done

done

sleep 15

nvidia-smi
echo "All jobs completed."
