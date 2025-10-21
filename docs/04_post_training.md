# 🚧 Under construction!
# Post‑training: checkpoints and usage

This document explains how to run **post‑trained** world models (WMs) inside World‑in‑World while keeping the existing notes in this file. If you are using **zero‑shot** models, see the deployment instructions in `03_run_commands.md` and the environment table in `01_setup_env.md`.

## Concepts

- **Zero‑shot vs post‑trained.** Post‑trained WMs are fine‑tuned on additional data. In our system, post‑trained variants usually use the `FT` prefix in `<wm_type>` (e.g., `FTwan21`, `FTcosmos`).
- **Checkpoints.** Each WM wrapper should accept a path to a checkpoint (e.g., `--ckpt_path`) and load it during worker initialization.

## How to use a post‑trained WM

1. **Prepare the environment.**
   Install the WM’s environment as described in [01_setup_env.md: Environment for different WMs](01_setup_env.md#environment-for-different-WMs).

2. **Place checkpoints.**
   Store your fine‑tuned weights where the worker can read them (local path or mounted storage). Avoid placing them under the states directory.

3. **Configure worker arguments.**
   In `downstream/utils/workers_cfg.py`, add or edit the entry for your `<wm_type>` to include the checkpoint argument (e.g., `--ckpt_path=/abs/path/to/weights`).

4. **Select the correct `<wm_type>`.**
   Use the `FT`‑prefixed type when launching the WM manager. See examples in `03_run_commands.md` and the supported list in `downstream/vlm.py`.

5. **Launch and verify.**
   Start the WM manager with `num_workers=1`. Send a few requests to verify that the post‑trained weights are loaded and outputs differ from zero‑shot baselines as expected.

## Tips

- Keep a short README next to the checkpoint describing training data, steps, and license.
- Pin package versions in `downstream/api_models/env_config/*.txt` for reproducibility.
- When comparing zero‑shot vs `FT` models, use identical prompts and seeds if applicable.

---

**Original notes kept for reference (not removed):**

To support more diffusers model you can check the diffusers documentation (https://huggingface.co/docs/diffusers/en/api/pipelines) for more details.

Check `WM_server_usage_readme` and downstream/api_models/READMEs/model_template_README.md

After you add a new WM, update:
- wm_type [here](downstream/vlm.py#L27-L33).
- `downstream/utils/worker_manager.py#L752` to add the new WM to the worker manager.

---

[↩︎ Back to Getting Started Checklist](../README.md#1-checklist-for-running-an-evaluation)

---