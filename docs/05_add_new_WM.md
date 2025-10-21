# 🚧 Under construction!
# How to Add New WM to World-in-World

This guide explains how to register a new world model (WM) so that it can be managed by the WM server and used by the solvers. We keep the process minimal while preserving flexibility across different pipelines (Diffusers, custom, etc.).

## Quick reference

- Diffusers pipelines: see the official docs at https://huggingface.co/docs/diffusers/en/api/pipelines
- Check `WM_server_usage_readme` and `downstream/api_models/READMEs/model_template_README.md` for code templates.
- After adding a new WM, update:
  - `wm_type` list in [downstream/vlm.py](downstream/vlm.py#L27-L33)
  - the worker registry in `downstream/utils/worker_manager.py#L752`

## Step‑by‑step

1. **Create an inference wrapper.**
   Copy `downstream/api_models/READMEs/model_template_README.md` and implement a new model file under `downstream/api_models/`, e.g., `my_new_wm_model.py`. Expose a simple `infer(**kwargs)` function (or an equivalent class method) that:
   - loads the model once at init,
   - accepts the inputs needed by your solver (e.g., image(s), prompt, pose),
   - returns outputs in the expected format (images, videos, latents, or file paths).

2. **Define an environment spec.**
   Create `downstream/api_models/env_config/<your_model>.txt` listing the exact Python packages (and versions) needed. If your model uses Diffusers, try to reuse an existing environment file to reduce duplication.

3. **Register the WM type.**
   Add a new symbolic `wm_type` name and map it to your wrapper in:
   - [downstream/vlm.py](downstream/vlm.py#L27-L33) (for dispatch in agent/policy code),
   - `downstream/utils/worker_manager.py#L752` (so the WM manager can launch workers).

4. **Update worker configuration.**
   In `downstream/utils/workers_cfg.py`, fill the environment name, conda path, and the exact entry‑point command to launch your worker with your wrapper script.

5. **Test locally.**
   Start a WM manager with `num_workers=1` and send a few requests to confirm:
   - the model loads once per worker,
   - GPU memory remains stable across requests,
   - outputs are written to expected locations and returned correctly.

6. **Scale out.**
   Increase `num_workers` and use multiple GPUs via `CUDA_VISIBLE_DEVICES=...`. Confirm that jobs are load‑balanced and throughput increases as expected.

## Tips

- Prefer lazy loading for large checkpoints to reduce startup time.
- If your WM produces intermediate files (e.g., videos), ensure unique per‑request directories to avoid collisions.
- For post‑trained checkpoints, accept a `--ckpt_path` (or similar) argument and document it in your wrapper’s README.
- Reuse common utilities in `downstream/api_models/` (logging, timing, I/O) for consistency.

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
