# Deployment and evaluation command instructions

This document provides step‑by‑step commands for running evaluation, deploying world models (WMs), and deploying VLMs / 3D‑diffusion policy.

---

## VLM Deployment

In this section, we will show how to deploy VLM policy as a server for different tasks. We show how to deploy Qwen2.5‑VL‑72B‑Instruct‑AWQ and InternVL3‑78B‑AWQ as examples below. We use Qwen2.5‑VL‑72B‑Instruct‑AWQ as the default VLM in our experiments.

**Reminder:**
- You can deploy WM server, vLLM server, and evaluation script on different machines (remote deployment) to accelerate the evaluation and fully utilize GPU resources as long as they can communicate through network/ssh. In our experience, the compute bottleneck is usually the WM server, then the vLLM server, so you can deploy them on machines with more GPUs or better GPUs.

1. **Qwen2.5‑VL‑72B‑Instruct‑AWQ** — start a vLLM server:
```bash
conda activate vllm
CUDA_VISIBLE_DEVICES="0" bash downstream/scripts/vllm_dsai.sh 8000 &
```
Here `8000` is the vLLM server port. Adjust `CUDA_VISIBLE_DEVICES` as needed. For example, to use GPUs 1 and 2, set `CUDA_VISIBLE_DEVICES="1,2"`. If you have a single GPU with ≥80 GB memory, you can usually run one VLM per GPU.

2. **InternVL3‑78B‑AWQ** — start a vLLM server:
```bash
conda activate vllmnew
CUDA_VISIBLE_DEVICES="0" bash downstream/scripts/vllm_dsai_intern.sh 8000 &
```

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

## SAM2 Deployment

Deploy a SAM2 server for tasks that require segmentation (e.g., AR).
```bash
CUDA_VISIBLE_DEVICES="0,1" bash downstream/scripts/init_sam2_manager.sh <exp_id> <num_workers> &
```
- `CUDA_VISIBLE_DEVICES="0,1"`: GPUs used for SAM2.
- `<exp_id>`: experiment id for logs saved under `downstream/logs/<exp_id>/`.
- `<num_workers>`: number of SAM2 model instances (workers). The script will try to spread workers over the listed GPUs evenly.

**Ports:** default is `6001`. If `6001` is busy, the script increments the port until it finds an available one. You can change the base port in `downstream/scripts/init_sam2_manager.sh`.

Example:
```bash
CUDA_VISIBLE_DEVICES="0,1" bash downstream/scripts/init_sam2_manager.sh 08.29_expid 2 &
```


---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

## Grounding SAM2 Deployment

Deploy a Grounding SAM2 server for tasks that require grounding (e.g., AEQA).
```bash
CUDA_VISIBLE_DEVICES="0,1" bash downstream/scripts/init_gd_sam2_manager.sh <exp_id> <num_workers> &
```

**Port:** default is `6002`. You can change it in `downstream/scripts/init_gd_sam2_manager.sh`.

**Example:**
```bash
CUDA_VISIBLE_DEVICES="0,1" bash downstream/scripts/init_gd_sam2_manager.sh 08.29_expid 2 &
```

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

## World Model Deployment

**Prerequisites**
- Install the desired WM’s environment according to the specific WM repository instructions, as detailed in [01_setup_env.md: Environment for Different WMs](01_setup_env.md#environment-for-different-WMs).
- Fill the environment path, inference script path, and other required parameters for WM workers in `downstream/utils/workers_cfg.py`. Placeholders are provided.

**Reminder:**
1. You can deploy WM server, vLLM server, and evaluation script on different machines (remote deployment) to accelerate the evaluation and fully utilize GPU resources as long as they can communicate through network/ssh. In our experience, the compute bottleneck is usually the WM server, then the vLLM server, so you can deploy them on machines with more GPUs or better GPUs.

### WMs for Habitat‑sim tasks

Activate the vLLM environment to run the WM manager (we separate the WM worker env from the manager env for compatibility):
```bash
conda activate vllm
```

Initialize a WM manager with 4 workers on GPUs 0–3:
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" bash downstream/scripts/init_worldmodel_manager.sh <exp_id> <num_workers> <wm_type> &
```
- `<exp_id>`: log directory `downstream/logs/<exp_id>/`.
- `<num_workers>`: number of WM instances (workers). Workers are balanced across the listed GPUs.
- `<wm_type>`: WM type to deploy (see supported values in [downstream/vlm.py](downstream/vlm.py#L27-L33)).

**Port:** default is `7000`. If `7000` is busy, the script increments the port until it finds an available one.

**Example:**
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" bash downstream/scripts/init_worldmodel_manager.sh 08.09_AR_FTwan 4 FTwan22 &
```

### Zero‑shot vs post‑trained WM deployment

- Post‑trained WMs use the prefix `FT` in `<wm_type>` (e.g., `FTwan21`, `FTcosmos`).
- Zero‑shot WMs use the same name without the `FT` prefix.
- See the table in [01_setup_env.md: Environment for Different WMs](01_setup_env.md#environment-for-different-WMs) for each `<wm_type>` and whether it supports zero‑shot or post‑trained usage.
- Check the corresponding inference script to confirm input parameter formats and how to pass a path of your post‑trained weights.

**Related link**
- [05_WM_server_design](docs/05_WM_server_design.md) — design details of the WM server.

### WMs for Manipulation tasks

** For VLM planner: **
Initialize a WM manager with 4 workers on GPUs 0–3:
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" bash downstream/scripts/init_worldmodel_manager.sh <exp_id> <num_workers> <wm_type> --task_type=manipulation
```
- `<exp_id>`: log directory `downstream/logs/<exp_id>/`.
- `<num_workers>`: number of WM instances (workers). Workers are balanced across the listed GPUs.
- `<wm_type>`: WM type to deploy (see supported values in [downstream/vlm.py](downstream/vlm.py#L27-L33)).

** For diffusion planner: **
Initialize a WM manager with 4 workers on GPUs 0–3:
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" bash downstream/scripts/init_worldmodel_manager.sh <exp_id> <num_workers> <wm_type>
```
- `<exp_id>`: log directory `downstream/logs/<exp_id>/`.
- `<num_workers>`: number of WM instances (workers). Workers are balanced across the listed GPUs.
- `<wm_type>`: WM type to deploy (supported values are `igenex_manip` and `cosmos_manip`).

### WMs for other tasks
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" bash downstream/scripts/init_worldmodel_manager.sh <exp_id> <num_workers> <wm_type> --task_type=freetext &
```
Setting `--task_type=freetext` prevents the system from adding an extra template to the text input.

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

## Run the evaluation scripts


### General pattern

For all Habitat‑sim tasks, you need to activate the Habitat‑sim environment before running the evaluation scripts:
```bash
conda activate habitat025
```

```bash
CUDA_VISIBLE_DEVICES="0" bash downstream/scripts/init_solvers.sh <solver_module_name> <exp_id> <num_workers> "<vllm_hosts>" <WM_host> [extra_args...] &
```
- `<solver_module_name>`: e.g., `downstream.solver_AR`, `downstream.solver_IGNav`, `downstream.solver_AEQA`.
- `<exp_id>`: experiment state directory `downstream/logs/<exp_id>/` (we recommend including `_<wm_type>` in `<exp_id>` to form the final pattern `<Date>_<Task>_<wm_type>_<other_info>`, e.g., `09.11_AR_cosmos_debug1`).
- `<num_workers>`: number of evaluation processes to launch.
- `<vllm_hosts>`: one or more vLLM server addresses separated by commas.
- `<WM_host>`: WM server address which you have deployed in the previous steps.
- `[extra_args...]`: extra or task‑specific arguments.

### AR — example
```bash
CUDA_VISIBLE_DEVICES="0" bash downstream/scripts/init_solvers.sh downstream.solver_AR  09.11_AR_cosmos_debug1 4 "localhost:8000,localhost:8001,localhost:8002" localhost:7000 --sam2_host=localhost:6001 --use_WM &
```
Runs AR with 4 evaluation processes on GPU 0, vLLM at `localhost:8000,8001,8002`, WM at `localhost:7000`, and SAM2 at `localhost:6001`.

### IGNav — example
```bash
CUDA_VISIBLE_DEVICES="0" bash downstream/scripts/init_solvers.sh downstream.solver_IGNav 09.11_IGNav_FTwan21_debug1 4 "localhost:8000" localhost:7000 --use_WM &
```
Runs IGNav with 4 workers on GPU 0, vLLM at `localhost:8000`, and WM at `localhost:7000`.

### AEQA — example
```bash
CUDA_VISIBLE_DEVICES="0" bash downstream/scripts/init_solvers.sh downstream.solver_AEQA  09.12_AEQA_wan21_1 6 "localhost:8000" localhost:7000 --grounding_sam2_host=localhost:6002 --use_WM &
```
Runs AEQA with 6 workers on GPU 0, vLLM at `localhost:8000`, WM at `localhost:7000`, and Grounding SAM2 at `localhost:6002`.

### Baselines without a WM
Remove the `--use_WM` flag.

### Baselines (heuristic policy + WM)
Add the `--use_heur` flag. Example:
```bash
CUDA_VISIBLE_DEVICES="0" bash downstream/scripts/init_solvers.sh downstream.solver_IGNav 09.11_IGNav_FTwan21_debug1 4 "localhost:8000" localhost:7000 --use_WM --use_heur &
```

### Manip pattern

For manipulation tasks, use `conda activate wow-manip`. All the following commands should run under `downstream/world-in-world-manip`.

```bash
CUDA_VISIBLE_DEVICES="0" bash scripts/run_manip.sh \
    <solver_env> \
    <exp_id> \
    <model_name> \
    <num_workers> \
    <vllm_hosts> \
    <igenex_host> \
    [extra_args...] (optional)
```
- `<solver_env>`:
    - `vlm-base`: VLM planner without world model
    - `vlm-igenex`: VLM planner with world model
    - `diff-base`: diffuser actor without world model
    - `diff-igenex`: diffuser actor with world model
- `<model_name>`: name of VLM used by the VLM planner (e.g., `Qwen/Qwen2.5-VL-72B-Instruct-AWQ`)
- `<num_workers>`: number of parallel workers for evaluation
- `<vllm_hosts>`: hostname and port for vLLM server (same format as other tasks, e.g., `localhost:8010`)
- `<igenex_host>`: host for world model server (same format as other tasks, e.g., `localhost:6010`)

**Example** for `vlm-base` with `Qwen2.5-VL-72B-Instruct-AWQ`, exp_id `09.12_qwen_base`, 1 worker, vLLM at `localhost:8010`, and world model at `localhost:6010`:
```bash
CUDA_VISIBLE_DEVICES="0" bash scripts/run_manip.sh \
    vlm-base \
    09.12_qwen_base \
    Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
    1 \
    "localhost:8010" \
    "localhost:6010"
```

### Dataset generator example for manipulation tasks
The following command generates 10 episodes of demos for each variation (120 in total) of the `push_buttons` task and stores them in `./temp`:
```bash
python wiw_manip/envs/tools/dataset_generator_NLP.py --save_path "./temp" --tasks "push_buttons" --episodes_per_variation 10 --variations -1
```

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

### Common questions

**Q1. How do I set `vllm_host`, `WM_host`, and `sam2_host`?**
- If the server runs on the same machine as the solver, use `localhost:<port>` (e.g., `localhost:8000`).
- For a remote server, use the remote machine’s IP or hostname.
- `sam2_host` and `gd_sam2_host` must be able to access the states under `downstream/states/<exp_id>/`. In practice, run them on the same machine as the evaluation script.

**Q2. Can I use a remote proprietary VLM service (e.g., OpenAI) instead of self‑hosting?**
Yes. Change `planner_model` and `answerer_model` in `downstream/scripts/init_solvers.sh` and pass your API key in the CLI arguments.

**Q3. The WM server runs on a remote machine. The evaluation machine can ping it but the WM TCP port is blocked by firewall/permissions. What can I do?**

**Option A — Local port forwarding** (recommended if you can SSH from the client to the server).
On the **client machine**, after adding the client’s SSH public key to the server’s `~/.ssh/authorized_keys`, run:
```bash
ssh -fN -L 8010:localhost:8000 <user_on_server>@<server_hostname>
```
This opens `localhost:8010` on the **client** and forwards it to `localhost:8000` on the **server**. You can then access the WM server locally at `http://localhost:8000/`.

**Option B — Reverse SSH tunnel** (use only if the client cannot SSH to the server, but the server can reach the client).
This requires an SSH server running on the **client** and a reachable client (e.g., open inbound SSH or VPN). After adding the **server** key to the **client** `authorized_keys`, run on the **server**:
```bash
ssh -fN -R 8000:localhost:8010 <user_on_client>@<client_hostname>
```
This opens `localhost:8010` on the **server** and forwards it back to `localhost:8000` on the **client**. On the client side, you can then access the WM server locally at `http://localhost:8000/`.

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

## Get evaluation results

### Navigation tasks

After starting an evaluation, running states are saved under `downstream/states/<exp_id>/` and logs under `downstream/logs/<exp_id>/`.

To accumulate results across all workers (e.g., task success rate), from the repo root run:
```bash
PYTHONPATH=. python downstream/evaluator.py   <exp_id> --task <task_name>
```
**Examples:**
```bash
PYTHONPATH=. python downstream/evaluator.py   09.11_AR_cosmos_debug1 --task AR
PYTHONPATH=. python downstream/evaluator.py   09.12_AEQA_wan21_1 --task AEQA --openai_key=<your_openai_api_key>
PYTHONPATH=. python downstream/evaluator.py   09.11_IGNav_FTwan21_debug1 --task IGNav
```
For AEQA, we use GPT‑4o via the OpenAI API as the evaluator by default.

To only check already‑saved partial results (without re‑running), use:
```bash
PYTHONPATH=. python downstream/evaluator.py    <exp_id> --task <task_name> --only_check_exist
```
**Example:**
```bash
PYTHONPATH=. python downstream/evaluator.py   09.12_AEQA_wan21_1 --task AEQA --openai_key=<your_openai_api_key> --only_check_exist
```

### Manipulation tasks

After finishing an evaluation, results are saved under `running/<solver_env>/<model_name>/<exp_id>` by tasks. To aggregate results from all tasks, run:
```bash
python wiw_manip/aggregate_results.py running/<solver_env>/<model_name>/<exp_id>
```

**Example:**
```bash
conda activate wow-manip
python wiw_manip/aggregate_results.py running/vlm-base/Qwen2.5-VL-72B-Instruct-AWQ/09.12_qwen_base
```

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---