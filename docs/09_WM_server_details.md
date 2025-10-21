# 🚧 Under construction!
# WM Server Design and Usage

This document explains why we use a WM server, how it is structured, and how to operate it efficiently. It also preserves the original Q&A at the end.

## Why a server instead of calling inference scripts directly?

The WM server consists of two main parts:
- **WM manager**: tracks jobs, dispatches requests to workers, collects results.
- **WM workers**: long‑lived processes that run the WM inference script and hold model weights in memory.

This design collects jobs at the manager, dispatches them to available workers (load‑balanced by pending jobs), and aggregates results—achieving high throughput and good GPU utilization with minimal overhead.

## Request flow (high level)

1. The solver sends a generation request to the WM manager.
2. The manager enqueues the job and selects a worker based on queue length.
3. The worker performs inference (using your wrapper in `downstream/api_models/…`) and returns outputs.
4. The manager writes artifacts to disk and returns a structured response to the solver.

## Scaling and placement

- Use `CUDA_VISIBLE_DEVICES=...` to bind workers to specific GPUs.
- Increase `--num_workers` to scale out on a single machine.
- Run separate managers on multiple hosts if needed; the solvers accept a `WM_host` address.
- Keep the WM manager environment separate from the worker environment to avoid dependency conflicts.

## Ports and health

- Default WM manager port: **7000** (auto‑increments if busy).
- SAM2 default port: **6001**; Grounding SAM2 default port: **6002** (both can be changed in their scripts).
- Expose `/health` (if available) for basic liveness checks, or watch the logs in `downstream/logs/<exp_id>/`.

## Failure handling

- Workers are long‑lived; if a worker crashes, the manager will log the failure. Restart the worker or the manager as needed.
- Use unique per‑request output directories under `downstream/states/<exp_id>/` to avoid conflicts after retries.
- Keep checkpoint paths read‑only from workers; write results to task‑scoped directories.

## Security and networking

- When deploying across machines, restrict access at the firewall level to trusted hosts/subnets.
- If a port is blocked, use SSH port forwarding or reverse tunnels (see examples in `03_run_commands.md`).

---

## Common Questions (original content kept)

1) Why use a WM server instead of invoking the WM inference script directly?
    - WM server is composed of 2 main parts:
        - WM manager: responsible for managing the WM workers and dispatching jobs to them.
        - WM workers: responsible for running the WM inference script.
    - With this design, all jobs are first collected by the WM manager, and then be dispatched to different model instances in a load-balanced fashion (by count pending jobs), and the results will be collected and returned to the WM manager to achieve a high throughput and GPU utilization in a lightweight manner.

---

[↩︎ Back to Getting Started Checklist](../README.md#1-checklist-for-running-an-evaluation)

---