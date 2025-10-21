# Prepare Evaluation Datasets

In this document, we describe how to set up the evaluation datasets for our four evaluation tasks: AR, ImageNav (IGNav), AEQA, and Manipulation.

---

## Common Steps

Here we describe some common steps for all tasks, and then detailed task-specific instructions in sections below.

Create a `data/` folder in the project for storing datasets:
```bash
mkdir data
```

### For Habitat-sim based tasks (AR, IGNav, AEQA)
First you need to download the HM3D and MP3D scene datasets as described below.

#### HM3D (Habitat-Matterport 3D)

- Steps:
  1) you can download the dataset from the official page and instructions [here](https://github.com/matterport/habitat-matterport-3dresearch?tab=readme-ov-file) or use our pre-packaged bundle download at [here](https://drive.google.com/uc?export=download&id=1UsxvJpp7A4Byrd6E3Cjx6EoHcAph_rz5)
  2) Example command to download and unpack (at the project root ./world-in-world):
  ```bash
  gdown 'https://drive.google.com/uc?export=download&id=1UsxvJpp7A4Byrd6E3Cjx6EoHcAph_rz5' -O hm3d_val_with_configs.tar.gz
  tar -xf hm3d_val_with_configs.tar.gz
  ```
  3) The final scene dataset we use is the **val** split of HM3D, which final structure should be like:
     ```
     data/scene_datasets/hm3d/
     ├── val/
     │   ├── <scene-id-1>/...<assets>...
     │   ├── <scene-id-2>/...<assets>...
     │   └── ...
     ├── hm3d_basis.scene_dataset_config.json
     └── hm3d_annotated_basis.scene_dataset_config.json
     ```

#### MP3D (Matterport3D) for Habitat

- Steps:
  1) Follow the Habitat MP3D instructions [here](https://github.com/facebookresearch/Habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset) to obtain Habitat-ready .glb assets.
  2) Place scenes under data/scene_datasets/mp3d so each scene has <scene>/<scene>.glb:
     ```
     data/scene_datasets/mp3d/
     ├── 17DRP5sb8fy/...
     ├── 5ZKStnWn8Zo/...
     ├── ...
     └── mp3d.scene_dataset_config.json
     ```

### For RLbench based tasks (Manipulation)
- coming soon...

---

[↩︎ Back to Getting Started Checklist](../README.md#1-checklist-for-running-an-evaluation)

---

## Task-specific Instructions

### Download AR evaluation episodes

to download MP3D scenes as described above, and also download the AR evaluation episodes file.
Download our AR evaluation episodes (episodes_AR.json.gz) at [this link](https://huggingface.co/datasets/zonszer/WIW_datasets/blob/main/eval_datasets/AR/episodes_AR.json.gz) into `data/WIW_datasets/eval_datasets/AR/`.

```bash
mkdir -p data/WIW_datasets/eval_datasets/AR/
# Download and put the AR evaluation episodes into data/AR/
wget -c
https://huggingface.co/datasets/zonszer/WIW_datasets/blob/main/eval_datasets/AR/episodes_AR.json.gz -P data/WIW_datasets/eval_datasets/AR/
```

Then make sure the `data_dir` in the data loader is consistent with your download location:
In downstream/downstream_datasets.py:
```python
class ARDataset(TaskDataset):
    ...
    def __init__(
        self,
        subset=None,
        ...,
        data_dir="data/WIW_datasets/eval_datasets/AR/", # as default
    ):
```

### Download IGNav evaluation episodes

Two files of episodes dataset are needed, and **you do not need to unzip them manually**. The code handles unzip at runtime.

```bash
mkdir -p data/WIW_datasets/eval_datasets/IGNav/
# Download and put the IGNav evaluation episodes into data/IGNav/
wget -c https://huggingface.co/datasets/zonszer/WIW_datasets/blob/main/eval_datasets/IGNav/episodes_IGNav.json.gz -P data/WIW_datasets/eval_datasets/IGNav/
wget -c https://huggingface.co/datasets/zonszer/WIW_datasets/blob/main/eval_datasets/IGNav/igdataset_goal_imgs.zip -P data/WIW_datasets/eval_datasets/IGNav/
```

Expected layout (after the first run, the `goal_imgs_unzipped/` folder will be created automatically):

```
data/WIW_datasets/eval_datasets/IGNav/
├── goal_imgs_unzipped              # created automatically at runtime
├── igdataset_goal_imgs.zip
└── episodes_IGNav.json.gz
```

### Download AEQA evaluation episodes

Download our AEQA evaluation episodes (episodes_AEQA.json.gz) at [this link](https://huggingface.co/datasets/zonszer/WIW_datasets/blob/main/eval_datasets/AEQA/episodes_AEQA.json.gz) into `data/WIW_datasets/eval_datasets/AEQA/`.

```bash
mkdir -p data/WIW_datasets/eval_datasets/AEQA/
# Download and put the AEQA evaluation episodes into data/AEQA/
wget -c https://huggingface.co/datasets/zonszer/WIW_datasets/blob/main/eval_datasets/AEQA/episodes_AEQA.json.gz -P data/WIW_datasets/eval_datasets/AEQA/
```

Expected layout:

```
data/AEQA/
└── episodes_AEQA.json.gz
```

---

[↩︎ Back to Getting Started Checklist](../README.md#1-checklist-for-running-an-evaluation)

---