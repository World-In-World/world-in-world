# 🚧 Under construction!
# Post‑training: checkpoints and usage

This document explains how to run **post‑trained** world models (WMs) inside World‑in‑World. If you are using **zero‑shot** models, see the deployment instructions in `03_run_commands.md` and the environment table in `01_setup_env.md`.


## Collect data for post‑training
To post‑train a WM, you may need to generate your own data. Below we describe how to collect data for Habitat‑Sim–based tasks and RLBench‑based tasks.

### For Habitat‑Sim–based tasks (AR, IGNav, AEQA)

We use a different version of Habitat‑Sim than the default evaluation environment.

1) Create the conda environment:
```bash
conda env create --file downstream/api_models/env_config/train_svd.yaml
```
2) Install Habitat‑Sim:
```bash
conda install habitat-sim==0.3.2 withbullet headless -c conda-forge -c aihabitat
```
3) Activate the environment:
```bash
conda activate habitat032
```

4) Collect data using the helper script:
```bash
CUDA_VISIBLE_DEVICES="0" bash scripts/data_collect.sh <exp_id> <output_dir> <num_processes>
```
Arguments:
- `<exp_id>`: A short string to identify this collection run (e.g., a date or tag).
- `<output_dir>`: Directory where collected data will be saved.
- `<num_processes>`: Number of worker processes to use for rendering.

Example:
```bash
CUDA_VISIBLE_DEVICES="0" bash scripts/data_collect.sh 09.30_debug data/datasets__/09.30_debug 4
```
Additional options and defaults are documented in `scripts/data_collect.sh`. For detailed configuration flags, see `habitat_data/HabitatRender.py`. Accoding to our experience, the bottleneck is the IO speed and cpu cores, so u could run multiple processes (eg. 4) on the same GPU to speed up the data collection.

Note: The default turn angle is 22.5 degrees in `habitat_data/HabitatRender.py`. Some Habitat versions only support integer turn angles. If you hit an error, adjust the corresponding config type from integer to float in the Habitat config file (the error message points to the exact field).

### For RLBench‑based tasks (Manipulation)
- coming soon...


## Tips for post‑training

**Zero‑shot vs. post‑trained models**  
Post‑trained world models (WMs) are fine‑tuned on task‑specific or dom  ain‑specific data, whereas zero‑shot models rely solely on their pre‑training. In our codebase, post‑trained variants typically use the `FT` prefix in `<wm_type>` (e.g., `FTwan21`, `FTcosmos`).

**Best practices:**
- **Document your checkpoints:** Keep a short README alongside each checkpoint describing the training data, number of steps, hyperparameters, and license information.
- **Pin dependencies:** Record exact package versions in `downstream/api_models/env_config/*.txt` to ensure reproducibility across different environments.
- **Fair comparisons:** When comparing zero‑shot and post‑trained (`FT`) models, use identical prompts, random seeds, and evaluation settings to isolate the effect of fine‑tuning.


---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

**Original notes kept for reference (not removed):**

To support more diffusers models, see the diffusers documentation: https://huggingface.co/docs/diffusers/en/api/pipelines.

Check `WM_server_usage_readme` and downstream/api_models/READMEs/model_template_README.md

After you add a new WM, update:
- wm_type [here](downstream/vlm.py#L27-L33).
- `downstream/utils/worker_manager.py#L752` to add the new WM to the worker manager.

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---