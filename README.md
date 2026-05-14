<h1 align="center">
  MPU: Towards Secure and Privacy-Preserving Knowledge Unlearning for Large Language Models
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2602.23798">
    <img src="https://img.shields.io/badge/arXiv-2602.23798-b31b1b.svg?style=for-the-badge&logo=arxiv&logoWidth=20" alt="arXiv"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/Tristan0318/MPU/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/LICENSE-MIT-7799CC.svg?style=for-the-badge&logoWidth=20" alt="License MIT"></a>
</p>

<p align="center">
  <a href="https://openreview.net/profile?id=~Tiantong_Wang1">Tiantong Wang</a><sup>1,2</sup>,
  <a href="https://scholar.google.com/citations?user=QXkrWQoAAAAJ">Xinyu Yan</a><sup>1,2</sup>,
  <a href="https://scholar.google.com/citations?user=7YsN6lMAAAAJ">Tiantong Wu</a><sup>1,2</sup>,
  <a href="https://yuronghaoa.github.io/yuronghaoA/">Yurong Hao</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=DnEMIzYAAAAJ">Pengjun Xie</a><sup>3</sup>,
  <a href="https://sites.google.com/view/wyb/people">Wei Yang Bryan Lim</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>College of Computing and Data Science, Nanyang Technological University<br>
  <sup>2</sup>Alibaba-NTU Global e-Sustainability CorpLab (ANGEL)<br>
  <sup>3</sup>Alibaba Group
</p>

---

This repository contains the official code for the paper **"MPU: Towards Secure and Privacy-Preserving Knowledge Unlearning for Large Language Models"**.

**MPU** is an algorithm-agnostic privacy-preserving Multiple Perturbed Copies Unlearning framework for large language models. It addresses the dual non-disclosure constraint in which strict privacy requirements prohibit sharing either the server's parameters or the client's forget set, introducing two server-side modules: **Pre-Process** (distributes multiple perturbed and reparameterized model instances) and **Post-Process** (inverts reparameterization and aggregates updates via harmonic denoising).

---

## 🗺️ Navigation

- [📖 Overview](#-overview)
- [⚡ Installation](#-installation)
- [🗂️ Code Structure](#-code-structure)
- [📂 Data Preparation](#-data-preparation)
- [🧪 Running Experiments](#-running-experiments)
  - [Unlearning on TOFU](#unlearning-on-tofu)
  - [Unlearning on MUSE](#unlearning-on-muse)
  - [Baseline Execution](#baseline-execution)
  - [Configuration Details](#configuration-details)
- [🤝 Acknowledgement](#-acknowledgement)
- [🔖 Citation](#-citation)
- [📄 License](#-license)

---

## 📖 Overview

Machine unlearning for large language models often faces a privacy dilemma in which strict constraints prohibit sharing either the server's parameters or the client's forget set.

To address this dual non-disclosure constraint, we propose **MPU**, which primarily introduces two server-side modules:
- **Pre-Process:** The server distributes multiple perturbed and reparameterized model instances.
- **Post-Process:** The server inverts the reparameterization and aggregates updates with a harmonic denoising procedure to alleviate the impact of perturbation.

Our framework allows the client to execute unlearning locally on its private forget set without accessing the server's exact original parameters, while ensuring the server doesn't access the client's raw data.

---

## ⚡ Installation

The codebase requires Python >= 3.11.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tristan0318/MPU.git
   cd MPU
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or install locally using `setup.py` (with optional dev/eval extras):
   ```bash
   pip install -e .
   # pip install -e ".[lm-eval,dev]"
   ```

---

## 🗂️ Code Structure

```
configs/          Hydra configuration files for models, trainers, unlearning algorithms, and evaluation
scripts/          Shell scripts for running the MPU unlearning pipeline and evaluations
src/
├── train.py      Main entry point for the unlearning procedure
├── eval.py       Evaluation script
└── trainer/      Logic for different local unlearning methods and MPU update aggregation (pum_utils.py)
setup_data.py     Script to download required datasets and evaluation logs
```

---

## 📂 Data Preparation

We provide a script to download and prepare the evaluation data and logs for TOFU, MUSE, and other related benchmarks:

```bash
python setup_data.py --eval_logs --idk --wmdp
```

| Flag | Description |
|---|---|
| `--eval_logs` | Downloads TOFU and MUSE evaluation logs (retained and finetuned models) |
| `--idk` | Downloads the IDK dataset |
| `--wmdp` | Downloads and extracts the WMDP corpora |

---

## 🧪 Running Experiments

All experiment configurations are managed using Hydra and are located in the `configs/` directory. Bash scripts in `scripts/` are provided to reproduce the experiments from the paper.

### Unlearning on TOFU

```bash
bash scripts/mpu_tofu.sh
```

### Unlearning on MUSE

```bash
bash scripts/mpu_muse.sh
```

### Baseline Execution

Scripts to run the baseline single-copy unlearning or fine-tuning without the MPU framework:

```
scripts/tofu_unlearn.sh
scripts/muse_unlearn.sh
scripts/tofu_finetune.sh
```

### Configuration Details

Inside the bash scripts, you can customize various hyperparameters for MPU:

| Parameter | Description |
|---|---|
| `PUM_M_LIST` | Number of perturbed copies ($m$) |
| `PUM_KAPPA` | Noise scale ($\kappa$) |
| `PUM_REPARAM` | Whether to enable function-preserving reparameterization |
| `client_methods` | Local unlearning algorithm (`NPO`, `DPO`, `GradAscent`, `SimNPO`, `UnDIAL`, `SatImp`) |

---

## 🤝 Acknowledgement

This paper is inspired by [OpenUnlearning](https://github.com/locuslab/open-unlearning), and also developed based on OpenUnlearning. Our implementation is built upon the [TOFU](https://github.com/locuslab/tofu) and [MUSE](https://github.com/swj0419/muse_bench) benchmarks.

---

## 🔖 Citation

If you find this work useful, please kindly consider citing our paper:

```bibtex
@misc{wang2026mpusecureprivacypreservingknowledge,
      title={MPU: Towards Secure and Privacy-Preserving Knowledge Unlearning for Large Language Models},
      author={Tiantong Wang and Xinyu Yan and Tiantong Wu and Yurong Hao and Pengjun Xie and Wei Yang Bryan Lim},
      year={2026},
      eprint={2602.23798},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.23798},
}
```

---

## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for details.
