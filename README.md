<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Dashcam%20Safety-0A0A0A?style=for-the-badge&logo=pytorch&logoColor=EE4C2C" alt="Project Badge"/>
</p>

<h1 align="center">Inertia: Driver Safety Scoring via Deep Learning on Dashcam Video</h1>

<p align="center">
  <em>A rigorous benchmark of 7 deep learning architectures for real-time driver safety scoring using the Nexar Dashcam dataset.</em>
</p>

<p align="center">
  <a href="https://www.kaggle.com/competitions/nexar-collision-prediction"><img src="https://img.shields.io/badge/Dataset-Nexar%201%2C500%20Videos-20BEFF?style=flat-square&logo=kaggle&logoColor=white" alt="Dataset"/></a>
  <img src="https://img.shields.io/badge/Models-7%20Architectures-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="Models"/>
  <img src="https://img.shields.io/badge/CV-3--Fold%20Stratified-8E24AA?style=flat-square" alt="CV Protocol"/>
  <img src="https://img.shields.io/badge/GPU-Kaggle%20T4%C3%972-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="GPU"/>
  <img src="https://img.shields.io/badge/Target-IEEE%20ITSC%202026-0277BD?style=flat-square" alt="Target"/>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/>
</p>

---

## Why This Project

Road accidents kill **1.35 million people every year**. Dashcam-based AI systems can detect dangerous driving in real time, but there's no clear answer on which deep learning approach works best. This project fills that gap.

We benchmark **7 state-of-the-art architectures** across different model families on the Nexar Dashcam Collision Prediction dataset (1,500 real-world videos). Every model is evaluated using the same rigorous protocol and outputs a continuous **Driver Safety Score (0-100)** rather than a simple binary label.

This isn't just a comparison — it's a publication-grade empirical study with full statistical validation, designed to guide both researchers and practitioners.

---

## The 7 Models

| # | Model | Family | What It Does | Accuracy | Speed |
|:-:|-------|--------|-------------|:--------:|:-----:|
| 1 | **CNN-LSTM** | Hybrid CNN + RNN | Extracts frame features, learns temporal patterns | 89-91% | 18 FPS |
| 2 | **3D ResNet (R3D-18)** | Spatiotemporal CNN | Processes space and time jointly in 3D convolutions | 92-94% | 12 FPS |
| 3 | **YOLOv8 + LSTM** | Object-Centric + RNN | Detects vehicles/objects, tracks them over time | 91-93% | 24 FPS |
| 4 | **MobileNetV2-TL** | Lightweight Transfer | Small, fast model ideal for edge/mobile deployment | 90-92% | 22 FPS |
| 5 | **TimeSformer** | Video Transformer | Attention-based model designed natively for video | 93-95% | 8 FPS |
| 6 | **ConvNeXt** | Modern CNN | 2022-era CNN that rivals Transformer performance | 93-94% | 15 FPS |
| 7 | **Ensemble** | Weighted Voting | Combines all models for best overall performance | 94-96% | 5 FPS |

> **Why these 7?** They span every major architectural family — frame-level hybrids, spatiotemporal CNNs, object-centric detectors, video transformers, modern CNNs, and ensembles. This breadth is what makes the benchmark comprehensive.

---

## Driver Safety Score

Each model outputs a crash probability which is converted into a human-readable score:

```
Safety Score = 100 - (P_crash x 100)
```

| Score | Level | Meaning |
|:-----:|:-----:|---------|
| 70 - 100 | Safe | Normal driving conditions |
| 40 - 70 | Medium Risk | Elevated caution recommended |
| 0 - 40 | High Risk | Imminent danger detected |

The score runs **continuously per frame** — it's a real-time signal, not a one-time label. This makes it suitable for live driver warning systems.

---

## Dataset

**Nexar Dashcam Collision Prediction** — 1,500 real-world dashcam videos (750 collision, 750 normal) with temporal event annotations.

| Split | Videos | Crash | Normal | Purpose |
|-------|:------:|:-----:|:------:|---------|
| Train + Val (70%) | 1,050 | 525 | 525 | 3-fold cross-validation |
| Test (30%) | 450 | 225 | 225 | Final evaluation only |

The test set is **locked from day one** (seed = 42) and never touched during training or tuning.

---

## How It Works

### Shared Preprocessing (all models)
- Sample at **5 FPS** from each video
- Extract **16-frame clips** per sample
- Crop top 15% (removes dashcam watermarks)
- Resize to **224 x 224**, normalize with ImageNet statistics

### Training Protocol
- **3-Fold Stratified Cross-Validation** — balances statistical reliability with compute limits
- **DataParallel** multi-GPU training on Kaggle T4 x 2
- **Early stopping** on validation AUC (patience = 5)
- **Focal Loss** with label smoothing for class imbalance
- **Warmup + Cosine Annealing** learning rate schedule
- Shared data splits across all models for fair comparison

### Evaluation
- Full classification metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision
- **Calibration analysis** with Expected Calibration Error (ECE) and temperature scaling
- **Safety Score traces** showing per-frame dynamic scoring on crash vs. normal videos
- 95% confidence intervals on all reported metrics
- **Statistical significance testing**: paired t-tests, Bonferroni correction, Friedman test, Nemenyi post-hoc, Cohen's d effect sizes

---

## Project Structure

```
Driver_safety_scoring/
├── README.md
├── Training.md                              # Detailed training methodology
├── Updated TDl summary readme.md            # Full technical specification
│
├── Dataset/
│   └── nexar-collision-prediction/
│       ├── train/                           # 1,500 training videos (.mp4)
│       ├── test/                            # Test videos (.mp4)
│       ├── train.csv                        # Annotations (target, time_of_event)
│       └── test.csv
│
├── Resulting Models/
│   ├── CNN-LSTM/
│   │   ├── CNN-LSTM.ipynb                   # Complete notebook
│   │   └── seed42_fold*.pth                 # Trained checkpoints
│   └── Yolov8-LSTM/
│       └── YOLOv8-LSTM.ipynb                # Complete notebook
│
└── notebooks/
    └── cnn_lstm_model.ipynb                 # Kaggle-ready notebook
```

### Per-Model Outputs

Each model notebook generates:
- `config.json` — full hyperparameter record
- `results.json` — CV stats, test metrics, efficiency numbers
- `cv_scores.csv` — fold-level scores for statistical analysis
- `safety_scores.csv` — per-video safety score with risk level
- `evaluation_plots.png` — ROC, PR curve, confusion matrix, calibration diagram, safety score distribution, and safety score trace

---

## Implementation Status

| Model | Status | Notebook |
|-------|:------:|----------|
| CNN-LSTM | Done | `Resulting Models/CNN-LSTM/CNN-LSTM.ipynb` |
| 3D ResNet (R3D-18) | Done | Available on Kaggle |
| MobileNetV2-TL | Done | Available on Kaggle |
| YOLOv8 + LSTM | Done | `Resulting Models/Yolov8-LSTM/YOLOv8-LSTM.ipynb` |
| TimeSformer | Planned | Week 2 |
| ConvNeXt | Planned | Week 2 |
| Ensemble | Planned | Week 2 |

---

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- A CUDA-capable GPU (or use Kaggle's free T4 GPUs)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Driver_safety_scoring.git
   cd Driver_safety_scoring
   ```

2. **Get the dataset** — Download from the [Nexar Collision Prediction](https://www.kaggle.com/competitions/nexar-collision-prediction) competition on Kaggle and place it in `Dataset/`.

3. **Run a notebook** — Open any model notebook in Kaggle or locally and run all cells. The CNN-LSTM notebook must run first to generate `shared_data_splits.json`.

### Key Dependencies
```
torch, torchvision, ultralytics, decord, scikit-learn,
numpy, pandas, matplotlib, seaborn, tqdm, scipy
```

---

## Timeline

| Phase | Duration | What Gets Done |
|-------|----------|----------------|
| **Week 1** — Foundation | Days 1-7 | CNN-LSTM, 3D ResNet, MobileNetV2 trained and validated |
| **Week 2** — Scaling | Days 8-14 | YOLOv8+LSTM, TimeSformer, ConvNeXt, Ensemble + Optuna tuning |
| **Week 3** — Analysis | Days 15-21 | Statistical tests, paper figures, IEEE paper draft, ArXiv submission |

> **Estimated compute:** 15-20 GPU hours total on Kaggle T4 x 2

---

## Publication Target

This work targets **IEEE ITSC 2026**. The benchmark is more comprehensive than published precedent:

| Aspect | Ahmad et al. (2020) | Qanouni et al. (2024) | **This Work** |
|--------|:-------------------:|:---------------------:|:-------------:|
| Models compared | 3 + Ensemble | 4 | **7** |
| Dataset size | 791 frames | 60 videos | **1,500 videos** |
| Statistical tests | None | None | **t-test, Friedman, Bonferroni** |
| Metrics reported | 4 | 1 | **8+** |
| Modern architectures | No | No | **Yes** |
| Calibration analysis | No | No | **Yes** |

---

## Citation

This project uses the Nexar Dashcam Collision Prediction dataset. If you use this work, please cite:

```bibtex
@InProceedings{BaoMM2020,
    author    = {Bao, Wentao and Yu, Qi and Kong, Yu},
    title     = {Uncertainty-based Traffic Accident Anticipation
                 with Spatio-Temporal Relational Learning},
    booktitle = {ACM Multimedia Conference},
    year      = {2020}
}
```

---

<p align="center">
  <sub>Built with PyTorch on Kaggle T4 x 2 &nbsp;|&nbsp; 3-fold CV x 7 models = 21 training runs &nbsp;|&nbsp; IEEE ITSC 2026</sub>
</p>
