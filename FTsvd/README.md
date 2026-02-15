## FTsvd: Finetuning Stable Video Diffusion (SVD) with Action Conditioning

This folder contains scripts to finetune SVD with action conditioning and to run inference workers for navigation/manipulation tasks.
SVD† below refers to our finetuned SVD variant.

### Environment setup

```bash
# 1) Create env
conda create -y -n FTsvd python=3.9 cmake=3.14.0
# 2) (Optional) Habitat-Sim, only needed if you use Habitat-based data/tools
conda install habitat-sim==0.3.2 withbullet headless -c conda-forge -c aihabitat
# 3) Install training deps (run from repo root)
pip install -r FTsvd/train_svd.txt
# 4) Install our self-modified version of diffusers (required)
pip install -e FTsvd/diffusers-private
```

Notes:
- We use `pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121` to install PyTorch and related packages. # TODO: check if necessary

### Pretrained SVD checkpoints

- Google Drive (all SVD† checkpoints, navigation + manipulation): `ckpts_upload.tar.gz`  
  Download:
  ```bash
  pip install gdown
  gdown 'https://drive.google.com/uc?export=download&id=160vRcbQinlqD_SOrNyKpkIrQWpYWpgcf' -O ckpts_upload.tar.gz
  ```

  Extract into `FTsvd/ckpts_upload/...` from repo root:
  ```bash
  mkdir -p FTsvd
  tar -xzvf ckpts_upload.tar.gz -C FTsvd
  ```
  After extracting, you should have `FTsvd/ckpts_upload/...`.

- Hugging Face Hub (mirror hosting the same artifacts):
  Browse: `https://huggingface.co/datasets/zonszer/WIW_ckpts/tree/main`

  Python (optional) download example:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download(
      repo_id="zonszer/WIW_ckpts",
      repo_type="dataset",
      local_dir="FTsvd/ckpts_upload",
      local_dir_use_symlinks=False,
  )
  ```

## Finetuning SVD

We finetune the SVD UNet with additional action conditioning for:
- navigation (`--action_input_channel` commonly 14)
- manipulation (`--action_input_channel` depends on your dataset; often 10)

Training entrypoint is `FTsvd/train_svd.py`. Note by default we use 4 H100 GPUs (95GB VRAM each) to 
finetune the model.

### 1) Prepare the pretrained SVD checkpoint

`--pretrained_model_name_or_path` must point to either:
- a local folder containing subfolders: `feature_extractor/`, `image_encoder/`, `vae/`, `unet/`
- or a Hugging Face model id (will be fetched into HF cache if available)

Example local snapshot path (also referenced in `scripts/training_dsai_acm_nav.sh`):
`FTsvd/ckpts_upload/models--stabilityai--stable-video-diffusion-img2vid`

### 2) Prepare the finetuning dataset

- `--base_folder` / `--val_base_folder` accept one or more dataset roots (space-separated).
- Expected pattern under each dataset root:
  - `<root>/*/traj-*/waypoint-*/metadata.json`
  - Each `waypoint-*` folder contains frame PNGs whose filenames include `type-rgb` and an index like
    `<anything>-<frame_idx>_<...>type-rgb<...>.png` (the loader sorts using `<frame_idx>`).
- This dataset format is produced by `scripts/data_collect.sh`.

Or you can download and use a small example finetuning dataset (navigation) as a reference:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download zonszer/WIW_example_dataset \
  --repo-type dataset \
  --local-dir data/example_dataset \
  --local-dir-use-symlinks False \
  --include "example_dataset/**"
```
This will create `data/example_dataset/example_dataset/...`.

### 3) Launch finetuning

```bash
bash FTsvd/train_svd.sh
```

Tip: make sure `config/accelerate_deepspeed_o1_config.yaml` has `num_processes` equal to the number of GPUs in `CUDA_VISIBLE_DEVICES`.

Resume:
- `--resume_from_checkpoint latest` to pick the newest `checkpoint-<step>` under `--output_dir`
- or `--resume_from_checkpoint /abs/path/to/checkpoint-<step>`

### 4) Key arguments (quick reference)

- **Data / IO**
  - `--base_folder`, `--val_base_folder`: one or more dataset roots (space-separated)
  - `--output_dir`: run directory; checkpoints saved under `output_dir/checkpoint-<global_step>`
- **Video shape**
  - `--num_frames`: number of target frames per sample
  - `--width`, `--height`: training resize
- **Checkpointing**
  - `--checkpointing_steps`: save training state every N global steps
  - `--checkpoints_total_limit`: keep only the latest N checkpoints
  - `--resume_from_checkpoint`: `latest` or a specific checkpoint path
- **Action conditioning**
  - `--task_type`: `navigation` or `manipulation`
  - `--action_input_channel`: match your dataset (navigation often 14; manipulation often 10)
  - `--action_strategy`: `micro_cond` (default) or others supported by the code
- **What gets trained (UNet)**
  - `--train_param_type`: e.g., `full` to train all UNet parameters

## Inference (world model server)

Launch the world model manager (navigation):
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" \
bash downstream/scripts/init_worldmodel_manager.sh 08.09_XXX 4 igenex --task_type=navigation &
```

Manipulation:
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" \
bash downstream/scripts/init_worldmodel_manager.sh 08.09_XXX 4 igenex_manip --task_type=manipulation &
```

- Configure worker commands (Python executable, `unet_path`, `svd_path`, sizes, etc.) in:
  - `downstream/utils/workers_cfg.py` (see the `COMMON_ARGS` dictionary for your platform)
- If you need Hugging Face access during inference, set your token in:
  - `downstream/scripts/set_env_variable.sh` (set `HF_TOKEN=...`)

