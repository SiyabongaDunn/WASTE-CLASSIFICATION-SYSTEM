# WASTE-CLASSIFICATION-SYSTEM

**Automated Waste Classification using Vision Transformers vs CNNs with Explainable AI - SortLight Deployment**

[Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
[TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-orange)
[Streamlit 1.35](https://img.shields.io/badge/Streamlit-1.35-red)
[License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**Authors:** Siyabonga Dunn (223017814) & Alunge Hlubi (223019985)  
**Supervisor:** Dr. Andronicus Ayobami Akinyelu  
**Institution:** University of KwaZulu-Natal | School of Mathematics, Statistics and Computer Science  
**Academic Year:** 2025/2026

> **Deployment Status:** ✅ ViT Loaded Successfully | `localhost:8501` | Demo: Hennessy Box → 84.2% Cardboard

---

## 1. Research Overview

Global municipal waste is projected to reach **3.4 billion tonnes by 2050** (World Bank). South Africa generates ~122M tonnes annually, with <10% recycled due to manual sorting inefficiency.

Existing sorting systems use CNNs that classify by **local texture** - patch by patch. This fails on visually similar waste like clear **Glass vs PET Plastic** or crumpled **Paper vs Cardboard**.

**Research Question:** *Does global self-attention (ViT) outperform local convolution (CNN) for recyclable waste classification?*

We implement and compare 3 models on 6 recyclable categories:

| Model | Architecture | Params | Input | Best Test Accuracy | F1-Macro | Inference Time |
|---|---|---|---|---|---|---|
| **ViT-base-patch16-224** | Transformer (12 layers, 12 heads, 768 dim) | ~86M | 224x224 | **94.3%** | **0.9145** | ~210ms |
| ResNet50V2 | Deep Residual CNN | ~25M | 224x224 | 91.1% | 0.889 | ~95ms |
| MobileNetV2 | Lightweight CNN (Depthwise Separable) | ~3.4M | 224x224 | 89.4% | 0.872 | ~32ms |

ViT trained with AdamW (lr=3e-4), weight decay 0.05, label smoothing 0.1, RandAugment, early stopping, 50 epochs.

### Why This Matters for SA
- Automates recycling at source (households, campus bins)
- Reduces contamination in recycling streams
- Lightweight MobileNetV2 version for edge devices (Raspberry Pi sorting bin)

## 2. Dataset

**Kaggle - Garbage Classification Dataset**  
Source: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification  
Total: **2,527 images** across 6 classes

| Class | Count | % | Challenge |
|---|---|---|---|
| Cardboard | 403 | 15.9% | Boxes, packaging, Hennessy box |
| Glass | 501 | 19.8% | Clear bottles vs plastic confusion |
| Metal | 410 | 16.2% | Cans, foil, tins |
| Paper | 594 | 23.5% | Newspaper, office paper, magazines |
| Plastic | 482 | 19.1% | PET bottles, bags, containers |
| Trash | 137 | 5.4% | Non-recyclable, most confused, imbalanced |

**Preprocessing (`utils.py`):**
- Resize to 224x224, normalization to [-1, 1]
- Stratified split: 70% Train / 15% Validation / 15% Test
- Augmentation: rotation ±20°, horizontal flip, zoom 0.2, color jitter
- Class imbalance handled via class weights

> `data/` folder is gitignored per UKZN policy (large binaries). Download original Kaggle set and extract to `data/`.

## 3. SortLight - Real-Time Deployment System

This repo includes **SortLight**, a Streamlit-based deployable frontend for real-time inference - not just training notebooks.

**Pipeline:**
```
Camera / Upload → Streamlit (app.py) → WasteClassifier (inference.py) → ViT/CNN Model → Prediction + Confidence + XAI Heatmap → User
```

**Features:**
- Drag-and-drop or live webcam input
- Model selector: ViT / ResNet50V2 / MobileNetV2
- Confidence bar chart for all 6 classes
- Top-3 predictions display
- **Explainable AI (XAI) Toggle:**
  - ViT: Attention Rollout + Chefer LRP (shows which patches model looked at - used in Hennessy Box demo)
  - CNN: Grad-CAM (last conv layer heatmap)
- Runs on CPU (no GPU needed for inference)

**Files:**
- `app.py` - Main Streamlit app UI logic
- `inference.py` - `WasteClassifier` class with auto-download fix
- `vit_model.py` - Custom ViT with `_attn_scores` exposed for XAI
- `sortlight_component.py` / `sortlight_frontend/` - Custom CSS/JS, SortLight branding
- `launcher.py` / `main.py` - Training entry points
- `utils.py` - `preprocess_image()`, `class_names` list

## 4. Installation & Quick Start (No Retraining Required)

This is the critical fix for `Available models: []`. Weights are **not** stored on GitHub (blocked by `.gitignore` - `*.h5` >100MB and GitHub 100MB limit). They are stored on Google Drive and auto-downloaded on first run via `gdown`.

### Steps:

```bash
# 1. Clone
git clone https://github.com/SiyabongaDunn/WASTE-CLASSIFICATION-SYSTEM.git
cd WASTE-CLASSIFICATION-SYSTEM

# 2. Create venv (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run - first run will download ~330MB model automatically to models/
streamlit run app.py
```

Open http://localhost:8501

**What happens automatically in `inference.py`:**
```python
import gdown
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vit_weights.weights.h5')
DRIVE_URL = "https://drive.google.com/uc?id=1G6fg3M7USlyPf8ACqxkcoRHv7g8r8Erl"

if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    print(f"Model not found, downloading from Drive to {MODEL_PATH}...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
```
This fixed the `Available models: []` bug. `.gitignore` correctly blocks `*.h5`, `__pycache__/`, `data/`, `*.pyc`.

> **UKZN Lab Note:** If `pip install` is blocked by Application Control Policy, the app will still run if `models/vit_weights.weights.h5` already exists in `models/` folder (copy it via USB).

## 5. Training (Optional - Reproduce Results)

If you want to retrain from scratch:

```bash
# Check config.yaml for hyperparameters
python main.py --model vit --epochs 50 --batch_size 32 --lr 3e-4
python main.py --model resnet50v2 --epochs 40
python main.py --model mobilenetv2 --epochs 40
```

- Best checkpoint saved to `models/` as `vit_weights.weights.h5`, `resnet_weights.weights.h5`, etc.
- Logs, accuracy/loss curves saved to `logs/`
- Confusion matrices saved to `notebooks/evaluation.ipynb`
- Configs editable in `config.yaml`

## 6. Project Structure

```
WASTE-CLASSIFICATION-SYSTEM/
├── app.py                         # Streamlit main app - SortLight UI
├── inference.py                   # WasteClassifier with gdown auto-download fix (CRITICAL)
├── vit_model.py                   # Custom ViT-base-patch16-224 with attention scores exposed
├── utils.py                       # preprocess_image(), class_names = [cardboard, glass, ...]
├── sortlight_component.py         # Custom Streamlit component
├── launcher.py                    # Alternative launcher
├── main.py                        # Training loop (fit, evaluate)
├── config.yaml                    # Hyperparams: lr, epochs, batch, weight_decay
├── requirements.txt               # FIXED: includes gdown==4.7.1, tensorflow==2.20.0
├── .gitignore                     # Blocks data/, models/*.h5, __pycache__, *.pyc
├── README.md                      # This file
├── models/                        # GITIGNORED - auto-populates via gdown
│   ├── vit_weights.weights.h5     # ~330MB - Downloaded from Drive ID 1G6fg3M...
│   ├── resnet_weights.weights.h5  # (optional)
│   └── mobilenet_weights.weights.h5
├── data/                          # GITIGNORED - Kaggle dataset (2527 images)
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
├── notebooks/                     # EDA, training curves, evaluation
│   ├── eda.ipynb
│   ├── training_vit.ipynb
│   └── evaluation.ipynb
├── sortlight_frontend/            # Frontend assets for Streamlit
│   ├── style.css
│   └── components/
├── __pycache__/                   # GITIGNORED
└── .vscode/                       # VS Code settings
```

## 7. Results & Key Findings

### Quantitative:
- ViT outperforms CNNs by ~3.2% accuracy because self-attention captures **long-range shape** (box folds vs paper crumples) vs local texture.
- Paper (594 samples) easiest, Trash (137 samples) hardest due to imbalance.
- Most confusion: Trash → Plastic / Glass (visually similar transparent bags).

### Qualitative (XAI):
- ViT Attention Rollout on Hennessy Box (84.2% Cardboard): Focuses on edges, brown color, and Hennessy text logo - proves it learned shape semantics.
- ResNet Grad-CAM on same image: Focuses only on brown texture, misses box structure (71% confidence).
- MobileNetV2: Fastest (32ms) but confuses Paper vs Cardboard.

### Deployment Test:
| Test Image | ViT | ResNet50V2 | Ground Truth |
|---|---|---|---|
| Hennessy Box (brown) | 84.2% Cardboard | 71.3% Cardboard | Cardboard ✅ |
| Clear PET bottle | 91% Plastic | 78% Glass | Plastic ✅ |
| Crushed paper | 88% Paper | 85% Paper | Paper ✅ |

Full metrics, confusion matrices, attention maps in `notebooks/evaluation.ipynb` and `logs/`.

## 8. The Fix - Why `Available models: []` Happened

**Problem:** After pushing, app showed `Available models: []` on Streamlit.

**Root Cause:**
- `models/vit_weights.weights.h5` is 330MB > GitHub 100MB limit
- `.gitignore` has `*.h5` and `data/` - correct for repo hygiene
- So `git push` did NOT upload the model, only code
- `inference.py` looked for `models/vit_weights.weights.h5`, didn't find it, loaded 0 models

**Solution (Siyabonga Dunn - this commit):**
1. Added `gdown>=4.7.1` to `requirements.txt`
2. Modified `inference.py` to:
   - Check if `MODEL_PATH` exists
   - If not, auto-download from Google Drive `id=1G6fg3M7USlyPf8ACqxkcoRHv7g8r8Erl`
   - Then load weights
3. Now anyone cloning `main` can run `streamlit run app.py` without manual model placement

This satisfies the deployment requirement for UKZN final demo.

## 9. Requirements

```txt
gdown
tensorflow==2.20.0
keras==3.13.2
streamlit>=1.35.0
anyio>=4.0.0
starlette>=0.27.0
opencv-python>=4.8.0
matplotlib>=3.7.0
pillow>=10.4.0
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pyyaml>=6.0
tqdm>=4.65.0
```

- Python 3.12 (tested), 3.10+ should work
- Works on CPU, GPU optional for training

## 10. Acknowledgements & References

- Supervisor: Dr. Andronicus Ayobami Akinyelu - UKZN
- Dataset: Kaggle - Garbage Classification by asdasdasasdas
- ViT Paper: Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)
- ResNet: He et al. "Deep Residual Learning for Image Recognition"
- MobileNetV2: Sandler et al.
- XAI: Chefer et al. "Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers"
- Streamlit, TensorFlow, OpenCV communities

## 11. License & Citation

MIT License - Free for academic use.

If you use this work, please cite:

```bibtex
@misc{dunn_hlubi_2025_waste,
  author = {Dunn, Siyabonga and Hlubi, Alunge},
  title = {WASTE-CLASSIFICATION-SYSTEM: Vision Transformer vs CNN for Automated Waste Classification with SortLight Deployment},
  year = {2025},
  institution = {University of KwaZulu-Natal},
  supervisor = {Akinyelu, Andronicus Ayobami},
  url = {https://github.com/SiyabongaDunn/WASTE-CLASSIFICATION-SYSTEM}
}
```

---

**Maintainer:** Siyabonga Dunn - 223017814@stu.ukzn.ac.za  
**GitHub:** https://github.com/SiyabongaDunn/WASTE-CLASSIFICATION-SYSTEM  
**Branch:** `main` is stable and deployable - `streamlit run app.py`  
**Last Working Demo:** 84.2% Cardboard (Hennessy Box) - ViT loaded successfully


