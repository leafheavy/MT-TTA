<p align="right">
  <strong>English</strong> | <a href="README_cn.md">中文</a>
</p>

# Multi-Task Test-Time Adaptation
This is the official implementation of the paper Multi-Task Test-time Adaptation via Gradient Consensus and Plasticity Constraint, accepted at AAAI 2026.
| [View paper](Paper/CameraReady_v3.pdf)| [Appendix](Paper/Appendix_v3.pdf) |
| ------------------------------------  |  -------------------------------  | 

This repository also includes other multi-task test-time adaptation algorithms.
| Algorithms |
| ---------- |
|  M-TENT    |
|  M-EATA    |
|  M-SAR     |
|  M-ActMAD  |

## Definition
Multi-task test-time adaptation (MT-TTA) aims to adapt pre-trained models to dynamic environments during multi-task inference by leveraging unlabeled test data.


![The Framework of CoCo](framework_cmyk.jpg)

## Environment Configuration
Python Version: 3.10.13; CUDA Version: 12.8

| torch      | 2.7.0+cu118 | pypi_0 | pypi |
|------------|-------------|--------|------|
| torchaudio | 2.7.0+cu118 | pypi_0 | pypi |
| torchvision| 0.22.0+cu118| pypi_0 | pypi |

Tips: Please ensure that the Torch-related libraries and the CUDA version are compatible with each other.

Create a conda virtual environment named MT_TTA:
```bash
conda env create -f environment.yml -n MT_TTA
```

## Datasets
![The Corruption Validation on CelebA](Corruption.png)
### Dataset Acquisition
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [PlantData](https://pan.baidu.com/s/1wPhu7GjyMinLeDJbKGbFPw?pwd=kukn), which constructed based on [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) and [Rice Plant Disease Dataset](https://github.com/MHassaanButt/Rice-Disease-Classfication)
Tips: Please ensure that you appropriately cite both Plantvillage and Rice Plant Disease Dataset while using PlantData in your reaserch.
| Datasets   |  Total  |  Train  |  Test  | Tasks |
|------------|---------|-------- |--------| ----- |
| CelebA     | 202,599 | 141,819 | 60,780 |   4   |
| PlantData  | 19,219  | 13,453  |  5,766 |   3   |

### Dataset Processing
Please place the datasets in the /Datasetsfolder. Refer to the following structure:
```
MT_TTA
└── Datasets
    ├── PlantData
    │   ├── balanced_test.csv
    │   ├── balanced_train.csv
    │   ├── data.md
    │   ├── data_description_image
    │   └── pictures
    └── CelebA
        ├── identity_CelebA.txt
        ├── img_align_celeba.zip
        └── img_align_celeba
```

## Usage of Multi-Task Test-Time Domain Adaptation Algorithms

### M-TENT/EATA/SAR/CoCo
1. Modify the output file's name and storage path (lines 33 and 35).
2. Select the target dataset for adaptation (search for 'create_celeba' and 'create_plantdata').
3. Choose the domain adaptation algorithm and make the corresponding modifications as shown below.
4. Run the script: `python main.py`.
Tips: Hyperparameters can be specified via command-line arguments or by directly modifying the Hyper-parameters Controller in `utils.py`.

**M-TENT**
(1) Comment out the PC-related content (lines 62 and 63).
(2) Set `grad_adjust_fn` to `None`.
(3) Set `require_fishers` to `False`.

**M-EATA**
(1) Ensure the PC-related content is correct and not commented out (lines 62 and 63).
(2) Replace `coco_model = CoCo(...)` with:
```python
eata_model = EATA_MultiTask(mymodel, optimizer, fishers=fishers, fisher_alpha=args.PC_alpha, steps=1, episodic=False, d_margin=0.05)
```
(3) Replace `coco_model` with `eata_model`.

**M-SAR**
(1) Comment out the PC-related content (lines 62 and 63).
(2) Replace `coco_model = CoCo(...)` with:
```python
sar_model = SAR_MultiTask(mymodel, optimizer, steps=1, episodic=False)
```

(3) Replace `coco_model` with `sar_model`.

**CoCo**
(1) Ensure the PC-related content is correct and not commented out (lines 62 and 63).
(2) Set `grad_adjust_fn` to `gradient_consensus`.
(3) Set `require_fishers` to `True`.
Tips: Please refer directly to the `main.py` script for details.

### M-ActMAD
1. Modify the output file's name and storage path (lines 340 and 346).
2. Select the target dataset for adaptation (PlantData: lines 362-364; CelebA: lines 367-369).
3. Run the script: `python M-ActMAD.py`.
- Tips: Based on step 2, select the appropriate task names (line 343 or 344).

# Correspondence
Please contact Zhong Ye by zhongye0312 [at] gmail.com 📬.

# Citation
If our work is helpful in your research, please consider citing our paper:



