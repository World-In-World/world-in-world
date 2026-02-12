# Prepare the environment for evaluation

In this document, we outline the steps to set up the evaluation environment for our four tasks. Because we use two simulators for different tasks (Habitat‑sim for AR, ImageNav (IGNav), and AEQA; RLBench for Manipulation), we provide separate instructions for each.

---

## Environment for Habitat-sim

For Habitat‑sim installation, you can follow the official [Habitat‑sim](https://github.com/facebookresearch/Habitat-sim) documentation, or follow the steps below.

### Install Habitat‑sim v0.2.5
```bash
# Create and activate a clean env (for Habitat 0.2.5)
conda create -n habitat025 python=3.9 cmake=3.14 -y
conda activate habitat025

# Install habitat-sim v0.2.5 (Bullet + headless/EGL if running on a server)
conda install -y -c conda-forge -c aihabitat \
  habitat-sim=0.2.5 \
  headless \
  withbullet
```

### Install Habitat‑Lab v0.2.5 and other Python dependencies

The pins below match the Habitat 0.2.5 stack and CUDA 12.1 wheels for PyTorch 2.5.1.

```bash
# (from your project root)
mkdir -p src && cd src

#  pip install:
pip install \
  "numpy>=1.20,<1.24" \
  "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  --extra-index-url https://download.pytorch.org/whl/cu121

pip install --upgrade \
  "imageio[ffmpeg]" \
  "exceptiongroup>=1.2" \
  "pydantic>=2.7,<3" \
  pandas==2.2.3 \
  jaxtyping==0.2.36 \
  tiktoken==0.9.0 \
  anthropic==0.49.0 \
  json-repair \
  tabulate==0.9.0 \
  tenacity==9.0.0 \
  pyequilib==0.5.8 \
  open3d==0.18.0 \
  openai \
  gdown

# Pin Habitat-Lab and Habitat-Baselines to the commit we used before
pip install -e "git+https://github.com/facebookresearch/habitat-lab.git@094d6be2f9d057e4781a68ae792132895fd4d3d0#egg=habitat_lab&subdirectory=habitat-lab" \
            -e "git+https://github.com/facebookresearch/habitat-lab.git@094d6be2f9d057e4781a68ae792132895fd4d3d0#egg=habitat_baselines&subdirectory=habitat-baselines"
```

### Install the `open-eqa` subtree
```bash
cd ..
cd subtrees/open-eqa
pip install -e .
```

### Compatibility note (Habitat‑Baselines)

There is a bug in the latest `habitat_baselines` that requires adjusting the default cubemap projection size. Modify:

File:
[`src/habitat-baselines/habitat-baselines/habitat_baselines/common/obs_transformers.py`](src/habitat-baselines/habitat-baselines/habitat_baselines/common/obs_transformers.py#L807)

Change line 807:
```python
def get_cubemap_projections(
    img_h: int = 256, img_w: int = 256
) -> List[CameraProjection]:
```
to
```python
def get_cubemap_projections(
    img_h: int = 512, img_w: int = 512
) -> List[CameraProjection]:
```
where `512` matches the default depth image size in this codebase. If you need a different depth resolution or encounter any error, update these values accordingly.

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

## Environment for VLM deployment

We use vLLM to deploy VLMs as policy models for evaluation.

### Install vLLM

Note we use Qwen2.5‑VL‑72B‑Instruct‑AWQ as the default VLM in our experiments.

1. For **Qwen2.5‑VL‑72B‑Instruct‑AWQ**, we use:
```bash
# Create and activate a clean env (for vLLM)
conda create -n vllm python=3.9 -y
conda activate vllm

# Install vLLM and dependencies
pip install vllm==0.7.3 cloudpickle==3.1.1 dill==0.4.0
```
You can find our full package list at `downstream/api_models/env_config/vllm.txt`.

2. For **InternVL3‑78B‑AWQ**, we use:
```bash
# Create and activate a clean env (for vLLM)
conda create -n vllmnew python=3.11 -y
conda activate vllmnew

# Install vLLM and dependencies
pip install vllm==0.9.2 cloudpickle==3.1.1 dill==0.4.0
```
You can find our full package list at `downstream/api_models/env_config/vllmnew.txt`.

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

## Environment for SAM2 / Grounding SAM2 deployment
```bash
# Create and activate a clean env (for SAM2)
conda create -n sam2 python=3.10 -y
conda activate sam2
```

Then follow the official installation instructions of [SAM2](https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation). For the SAM2 environment we used, see `downstream/api_models/env_config/sam2.txt` for the full package list.

For Grounding SAM2, also install `ultralytics` in the same `sam2` env:
```bash
conda activate sam2
pip3 install ultralytics==8.3.118
```

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

## Environment for WIW-Manipulation

### Create conda environment and install PyTorch
```bash
cd downstream/world-in-world-manip

conda create -n wow-manip1 python=3.9 -y
conda activate wow-manip1
pip install setuptools==75.6.0 
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Install CoppeliaSim for physical simulation

Remember to copy the following export commands to your `.bashrc` or `.zshrc` and replace `$(pwd)` with the absolute path to `downstream/world-in-world-manip`.

```bash
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

### Install PyRep and AMSolver

You should be able to run evaluations with VLM proposer after this step.

```bash
cd wiw_manip/envs
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
cd ..
cp simAddOnScript_PyRep.lua $COPPELIASIM_ROOT
pip install -r requirements.txt
pip install -e .
cd ../..

pip install -r requirements.txt # other required packages
```

If you meet anything like `qt.qpa.plugin: Could not find the Qt platform plugin "xcb"`, try `pip uninstall opencv-python opencv-python-headless` and `pip install opencv-python-headless==4.11.0.86`

### Install 3D-Diffuser-Actor (for diff-base and diff-igenex)

```bash
mkdir src
cd src
git clone https://github.com/nickgkan/3d_diffuser_actor.git
cd 3d_diffuser_actor
pip install -e .
pip install openai-clip transformers==4.47.1 peft==0.11.1 diffusers==0.11.1 huggingface-hub==0.25.0
conda install dgl=2.4.0 -c dglteam/label/th24_cu121
cd ../..
```

### Configure the required ckpt files for 3D-Diffuser-Actor

Download our pretrained checkpoints for 3D-Diffuser-Actor:

```bash
pip install -U gdown

gdown "https://drive.google.com/uc?export=download&id=1QTKzDZvRUh3pVi-0ui1TT-CW7jlwZc6V" -O insert_onto_square_peg
gdown "https://drive.google.com/uc?export=download&id=1VZjtIEVdSVjpCYM824PKWTMXW6AcjQi8" -O push_buttons
gdown "https://drive.google.com/uc?export=download&id=1XHKYOMj2D5txC8LZBzkWU38Xjo1i3Pv7" -O slide_block_to_color_target
```

Then configure them in `downstream/world-in-world-manip/wiw_manip/configs/paths.py`.

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---

## Environment for different WMs

We create separate conda environments for different WMs to avoid dependency conflicts. Below are the environments used in our experiments for reference. Because many inference scripts use **diffusers**, it may be possible to reuse one environment across multiple WMs if dependencies are compatible.

| Model                                                                                         | Version                 | Pipeline  | Inference Script                                                                               | Env Config                                                                                                 | Setup Reference                                                                                                          | Notes                                                                |
| --------------------------------------------------------------------------------------------- | ----------------------- | --------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| [Cosmos-predict2 2B](https://github.com/nvidia-cosmos/cosmos-predict2)                        | zero-shot; post-trained | Diffusers | [`../downstream/api_models/cosmos_model.py`](../downstream/api_models/cosmos_model.py)               | [`../downstream/api_models/env_config/cosmos.txt`](../downstream/api_models/env_config/cosmos.txt)               | See setup in `cosmos_model.py` (lines 1–8): [L1–L8](../downstream/api_models/cosmos_model.py#L1-L8)                         | Env reused by **SVD** for zero-shot inference                        |
| [HunyuanVideo-I2V](https://github.com/Tencent/HunyuanVideo-I2V)                               | zero-shot               | Diffusers | [`../downstream/api_models/hunyuan_model.py`](../downstream/api_models/hunyuan_model.py)             | [`../downstream/api_models/env_config/hunyuan.txt`](../downstream/api_models/env_config/hunyuan.txt)             | See setup in `hunyuan_model.py` (lines 1–18): [L1–L18](../downstream/api_models/hunyuan_model.py#L1-L18)                    | —                                                                    |
| [LTX-Video 2B](https://github.com/Lightricks/LTX-Video)                                       | zero-shot; post-trained | Diffusers | [`../downstream/api_models/ltx_model.py`](../downstream/api_models/ltx_model.py)                     | [`../downstream/api_models/env_config/LTXvideo.txt`](../downstream/api_models/env_config/LTXvideo.txt)           | See setup in `ltx_model.py` (lines 1–9): [L1–L9](../downstream/api_models/ltx_model.py#L1-L9)                               | —                                                                    |
| [Wan2.1-I2V-A14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) | zero-shot               | Diffusers | [`../downstream/api_models/wan_model.py`](../downstream/api_models/wan_model.py)                     | [`../downstream/api_models/env_config/wan.txt`](../downstream/api_models/env_config/wan.txt)                     | See setup in `wan_model.py` (lines 1–5): [L1–L5](../downstream/api_models/wan_model.py#L1-L5)                               | —                                                                    |
| [Wan2.2-5B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)                 | zero-shot               | Diffusers | [`../downstream/api_models/wan22_ti2v_model.py`](../downstream/api_models/wan22_ti2v_model.py)       | `../downstream/api_models/env_config/wan22.txt`                                                               | See setup in `wan22_ti2v_model.py` (lines 7–12): [L7–L12](../downstream/api_models/wan22_ti2v_model.py#L7-L12)              | -                                          |
| [SVD-1.5B](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)                 | zero-shot               | Diffusers | [`../downstream/api_models/svd_model.py`](../downstream/api_models/svd_model.py)                     | *(same as Cosmos)*                                                                                         | —                                                                                                                        | Uses **Cosmos** env.                                                 |
| [Wan2.2-5B-diffsynth](https://github.com/modelscope/DiffSynth-Studio)                         | post-trained            | DiffSynth | [`../downstream/api_models/wan_model_diffsynth.py`](../downstream/api_models/wan_model_diffsynth.py) | [`../downstream/api_models/env_config/wan_diffsynth.txt`](../downstream/api_models/env_config/wan_diffsynth.txt) | See setup in `wan_model_diffsynth.py` (lines 1–5): [L1–L5](../downstream/api_models/wan_model_diffsynth.py#L1-L5)           | Base env for **Wan2.2-A14B-diffsynth** and **Wan2.1-14B-diffsynth**. |
| [Wan2.2-A14B-diffsynth](https://github.com/modelscope/DiffSynth-Studio)                       | post-trained            | DiffSynth | [`../downstream/api_models/wan_model_diffsynth.py`](../downstream/api_models/wan_model_diffsynth.py) | *(same as Wan2.2-5B-diffsynth)*                                                                            | —                                                                                                                        | Shares env with 5B.                                                  |
| [Wan2.1-14B-diffsynth](https://github.com/modelscope/DiffSynth-Studio)                        | post-trained            | DiffSynth | [`../downstream/api_models/wan_model_diffsynth.py`](../downstream/api_models/wan_model_diffsynth.py) | *(same as Wan2.2-5B-diffsynth)*                                                                            | —                                                                                                                        | Shares env with 5B.                                                  |
| [SE3DS](https://github.com/google-research/se3ds)                                             | zero-shot               | Custom    | [`../downstream/api_models/se3ds_model.py`](../downstream/api_models/se3ds_model.py)                 | [`../downstream/api_models/env_config/se3ds.txt`](../downstream/api_models/env_config/se3ds.txt)                 | Follow SE3DS README (Setup Instructions): https://github.com/google-research/se3ds?tab=readme-ov-file#setup-instructions | —                                                                    |
| [Pathdreamer](https://github.com/google-research/pathdreamer)                                 | zero-shot               | Custom    | [`../downstream/api_models/pathdreamer_model.py`](../downstream/api_models/pathdreamer_model.py)     | *(same as SE3DS)*                                                                                          | —                                                                                                                        | Shares env with **SE3DS**.                                           |
| [Navigation World Model](https://github.com/facebookresearch/nwm/)                            | zero-shot               | Custom    | [`../downstream/api_models/nwm_model.py`](../downstream/api_models/nwm_model.py)                     | [`../downstream/api_models/env_config/nwm.txt`](../downstream/api_models/env_config/nwm.txt)                     | Follow NWM README (Requirements): https://github.com/facebookresearch/nwm/?tab=readme-ov-file#requirements               | —                                                                    |
| [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)                      | post-trained            | Custom    | *to be released*                                                                               | *to be released*                                                                                           | *to be released*                                                                                                         | Placeholders in doc.                                                 |

---

[↩︎ Back to Getting Started Checklist](../README.md#2-checklist-for-running-an-evaluation)

---
