<!-- # World-in-World -->

<p align="center">
  <img src="./assets/logo.svg" width="330" alt="World-in-World logo"/>
</p>

<div align="center">
  <a href="https://world-in-world.github.io/"><img src="https://img.shields.io/badge/🌐 Website-Visit-slateblue" alt="Website badge"></a>
  <a href="https://arxiv.org/abs/2510.18135"><img src="https://img.shields.io/badge/arXiv-Abstract-orange" alt="arXiv badge"></a>
  <a href="https://huggingface.co/datasets/zonszer/WIW_datasets/tree/main"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Page-goldenrod" alt="HF badge"></a>
  <a href="https://github.com/World-In-World/world-in-world" target="_blank"><img src="https://img.shields.io/badge/GitHub-Repo-darkgray?style=flat&logo=github" alt="GitHub badge"></a>
<a href="https://world-in-world.github.io/subpages/index.html"><img src="https://img.shields.io/badge/Demo-Visit-slateblue" alt="Demo Badge"></a>
</div>

World-in-World is a unified **closed-loop** benchmark and toolkit for evaluating **visual world models (WMs)** by their **embodied utility** rather than only image or video appearance. World-in-World provides: (1) a unified online planning strategy that works with different WMs, (2) a unified action API that adapts to text, viewpoint, and low‑level controls, and (3) a task suite covering Active Recognition (AR), Active Embodied QA (A‑EQA), Image‑Goal Navigation (IGNav), and Robotic Manipulation.

---

## 📰 News
- **2025-10-22**: Preprint released on arXiv. Landing page and repository initialized.
- **2025-11-10**: Add post‑training instructions and data collection instructions in [data collection section](docs/04_post_training.md#collect-data-for-posttraining).
- **2026-02-09**: Add Manipulation task instructions and environment setup

---

## ✨ Overview

![Overview](assets/overview.png)

In this work, we propose World-in-World, which wraps generative <u>World</u> models <u>In</u> a closed-loop <u>World</u> interface to measure their practical utility for embodied agents. *We test whether generated worlds actually enhance embodied reasoning and task performance*—for example, helping an agent perceive the environment, plan and execute actions, and re-plan based on new observations *within such a closed loop*. Establishing this evaluation framework is essential for tracking genuine progress across the rapidly expanding landscape of visual world models and embodied AI.

---

## 🚧 Repository Status

The release will follow the to‑do list below and will be updated continuously.

**Under construction**
- Full documentation and tutorials for environment setup and task evaluation.
  - [X] AR, IGNav, AEQA
  - [X] Manipulation
- [X] WM post‑training instructions
- [ ] Instructions to add a new WM to World‑in‑World

---

## 🚀 Getting Started

### 1) Documentation structure

- [01_setup_env.md](docs/01_setup_env.md): Environment setup for all environments used in the repo.
- [02_evaluation_datasets.md](docs/02_evaluation_datasets.md): Datasets used for evaluation.
- [03_run_commands.md](docs/03_run_commands.md): How to deploy servers and run evaluation scripts.
- [04_post_training.md](docs/04_post_training.md): Post‑training configurations, data collection instructions, and checkpoints for different WMs.
- [05_add_new_WM.md](docs/05_add_new_WM.md): How to add a new WM to World‑in‑World.
- [09_WM_server_design.md](docs/09_WM_server_details.md): Design details of the WM server.

### 2) Checklist for running an evaluation

For any task, complete the following steps in order.

1. **Set up environments.**
   - **AR, IGNav, AEQA:** set up Habitat‑sim as described in [01_setup_env.md: Environment for Habitat‑sim](docs/01_setup_env.md#environment-for-Habitat-sim).
   - **Manipulation:** set up wow-manip as described in [01_setup_env.md: Environment for WIW-Manipulation](docs/01_setup_env.md#environment-for-WIW-Manipulation)
2. **Download scene datasets.**
   - **AR:** download MP3D as described in [02_evaluation_datasets.md: Common Steps](docs/02_evaluation_datasets.md#common-steps).
   - **IGNav, AEQA:** download HM3D as described in [02_evaluation_datasets.md: Common Steps](docs/02_evaluation_datasets.md#common-steps).
   - **Manipulation:** already exists in downstream/world-in-world-manip/wiw_manip/envs/vlm
3. **Download evaluation episodes.**
   - **AR:** see [02_evaluation_datasets.md: Download AR evaluation episodes](docs/02_evaluation_datasets.md#download-AR-evaluation-episodes).
   - **IGNav:** see [02_evaluation_datasets.md: Download IGNav evaluation episodes](docs/02_evaluation_datasets.md#download-ignav-evaluation-episodes).
   - **AEQA:** see [02_evaluation_datasets.md: Download AEQA evaluation episodes](docs/02_evaluation_datasets.md#download-AEQA-evaluation-episodes).
   - **Manipulation:** already exists in downstream/world-in-world-manip/data
4. **Deploy policies (VLM policy, heuristic policy, diffusion policy).**
   - **AR:** deploy VLM policy as in [03_run_commands.md: VLM Deployment](docs/03_run_commands.md#VLM-Deployment). If you use a heuristic policy, you can skip the VLM step.
   - **IGNav:** deploy VLM policy as in [03_run_commands.md: VLM Deployment](docs/03_run_commands.md#VLM-Deployment). If you use a heuristic policy, you can skip the VLM step.
   - **AEQA:** deploy VLM policy as in [03_run_commands.md: VLM Deployment](docs/03_run_commands.md#VLM-Deployment).
   - **Manipulation:** 
      - **VLM policy:** deploy the VLM policy as described in [03_run_commands.md: VLM Deployment](docs/03_run_commands.md#VLM-Deployment).
      - **Diffusion policy:** after configuring the ckpt paths in [01_setup_env.md: Configure the required ckpt files for 3D-Diffuser-Actor](docs/01_setup_env.md#configure-the-required-ckpt-files-for-3d-diffuser-actor), no additional deployment is needed.
5. **Deploy other task‑related models if needed.**
   - **AR:** deploy the SAM2 server as in [03_run_commands.md: SAM2 Deployment](docs/03_run_commands.md#SAM2-Deployment).
   - **IGNav:** no extra task models.
   - **AEQA:** deploy the Grounding SAM2 server as in [03_run_commands.md: Grounding SAM2 Deployment](docs/03_run_commands.md#Grounding-SAM2-Deployment).
   - **Manipulation:** no extra task models.
6. **Deploy the WM server.**
   - **AR, IGNav, AEQA:** see [03_run_commands.md: World Model Deployment](docs/03_run_commands.md#World-Model-Deployment) and [WMs for Habitat‑sim Tasks](docs/03_run_commands.md#WMs-for-Habitat-sim-Tasks).
   - **Manipulation:** see [03_run_commands.md: World Model Deployment](docs/03_run_commands.md#World-Model-Deployment) and [WMs for Manipulation Tasks](docs/03_run_commands.md#WMs-for-Manipulation-Tasks).
7. **Run the evaluation script.**
   - **AR, IGNav, AEQA:** see [03_run_commands.md: Run the Evaluation Scripts](docs/03_run_commands.md#Run-the-Evaluation-Scripts).
   - **Manipulation:** see [03_run_commands.md: Manip pattern](docs/03_run_commands.md#Manip-pattern).
8. **Accumulate results.**
   - **AR, IGNav, AEQA:** see [03_run_commands.md: Get evaluation results](docs/03_run_commands.md#Navigation-tasks).
   - **Manipulation:** see [03_run_commands.md: Get evaluation results](docs/03_run_commands.md#Manipulation-tasks).

After the first run, the environment and datasets are in place. For later runs, you usually only repeat **steps 4–8**.
If you encounter any issue, please feel free to open an issue or contact us.

P.S. if u have any question about the server deployment, you can also refer to [03_run_commands.md: Common questions](docs/03_run_commands.md#Common-questions) and [09_WM_server_design.md: WM Server Design Details](docs/09_WM_server_details.md#WM-Server-Design-Details) for troubleshooting.

---

## 🏆 Submit Custom Results to the Leaderboard

To submit new results to the leaderboard:

1. **Update this repository.**
   Fork/clone this repo, add your modifications (e.g., custom model inference script) and instructions for how we can reproduce the results, then open a pull request for review.

2. **Update the website leaderboard.**
   Fork/clone the website repo: https://github.com/World-In-World/World-In-World.github.io
   Edit `subpages/leaderboard.html`: https://github.com/World-In-World/World-In-World.github.io/blob/main/subpages/leaderboard.html
   Then open a pull request.

We will review the submission and, once verified, we will merge the changes and update the leaderboard accordingly.

---

## 📝 Citation

If you find this work useful, please cite:
```bibtex
@misc{zhang2025worldinworld,
  title        = {World-in-World: World Models in a Closed-Loop World},
  author       = {Zhang, Jiahan and Jiang, Muqing and Dai, Nanru and Lu, Taiming and Uzunoglu, Arda and Zhang, Shunchi and Wei, Yana and Wang, Jiahao and Patel, Vishal M. and Liang, Paul Pu and Khashabi, Daniel and Peng, Cheng and Chellappa, Rama and Shu, Tianmin and Yuille, Alan and Du, Yilun and Chen, Jieneng},
  year         = {2025},
  eprint       = {2510.18135},
  archivePrefix= {arXiv},
}
```
