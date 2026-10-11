# WASTE-CLASSIFICATION-SYSTEM

A deep learning research project comparing Vision Transformer (ViT) and Convolutional Neural Network (CNN) architectures for automated waste image classification across six recyclable material categories.

**Authors:** Siyabonga Dunn (223017814) & Alunge Hlubi (223019985)  
**Supervisor:** Dr. Andronicus Ayobami Akinyelu  
**Institution:** University of KwaZulu-Natal  
**Dissertation:** Vision Transformer Architecture for Enhanced Global Feature Extraction in Automated Waste Sorting

---

## Project Overview

Global waste generation is projected to reach 3.4 billion tonnes by 2050. Current automated sorting systems rely on CNNs which process images locally — patch by patch — and struggle to distinguish visually similar materials such as clear glass and PET plastic. This project investigates whether Vision Transformers, which model global spatial relationships across an entire image simultaneously, can provide a more robust and interpretable solution.

Three models are implemented, trained, and rigorously compared across three experimental configurations:

| Model | Type | Parameters |
|---|---|---|
| ViT-base-patch16-224 | Vision Transformer | ~86M |
| ResNet50V2 | CNN (deep residual) | ~25M |
| MobileNetV2 | CNN (lightweight) | ~3.4M |

---

## Results Summary

### Experiment 1 — Baseline Training

| Model | Accuracy | Macro F1 |
|---|---|---|
| MobileNetV2 | 83.8% | 0.819 |
| ResNet50V2 | 84.8% | 0.824 |
| **ViT Base-16** | **91.3%** | **0.885** |

### Experiment 2 — Targeted Augmentation (Trash class only)

| Metric | Baseline | Augmented |
|---|---|---|
| Accuracy | 91.3% | 89.7% |
| Macro F1 | 0.885 | 0.874 |
| Trash F1 | 0.67 | 0.70 |

### Experiment 3 — Class-Balanced Oversampling (Best Configuration)

| Metric | Score |
|---|---|
| Accuracy | **94.3%** |
| Macro F1 | **0.915** |
| Trash F1 | 0.70 |

### Statistical Validation — McNemar's Test

| Comparison | p-value | Result |
|---|---|---|
| ViT vs ResNet50V2 | p < 0.001 | ✅ Significant |
| ViT vs MobileNetV2 | p < 0.001 | ✅ Significant |
| ResNet50V2 vs MobileNetV2 | p = 0.596 | ❌ Not significant |

---

## Dataset

**Kaggle Garbage Classification Dataset**  
2,527 images across six waste categories:

| Class | Images |
|---|---|
| Cardboard | 403 |
| Glass | 501 |
| Metal | 410 |
| Paper | 594 |
| Plastic | 482 |
| Trash | 137 |



Source: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

---

## Project Structure

```
WASTE-CLASSIFICATION-SYSTEM/
│
├── notebooks/
│   └── Full_Pipeline(Latest version).ipynb   ← Complete training pipeline
│
├── src/
│   ├── data_loader.py        ← Dataset loading, augmentation, splits
│   ├── mobilenet.py          ← MobileNetV2 model definition
│   ├── resnet.py             ← ResNet50V2 model definition
│   ├── vit.py                ← ViT-base-patch16-224 model definition
│   ├── attention_maps.py     ← ViT attention maps + CNN Grad-CAM
│   ├── train.py              ← Training script for all three models
│   ├── evaluate.py           ← Metrics, confusion matrix, McNemar's test
│   └── predict.py            ← Single image prediction + explainability
│
├── app/
│   └── app.py                ← Streamlit web application
│
├── results/
│   ├── confusion_matrix_vit.png
│   ├── confusion_matrix_resnet.png
│   ├── confusion_matrix_mobilenet.png
│   ├── heatmap_gallery.png
│   ├── phase1_cv_results.csv
│   └── final_comparison_results.csv
│
├── presentations/
│   ├── Progress_Report_1.pptx
│   └── Progress_Report_2.pptx
│
├── requirements.txt
└── README.md
```

---

## Methodology

### Data Pipeline
- **Dataset split:** Stratified 80/20 train-test split — test set isolated immediately
- **Final training split:** 90/10 split within the 80% training pool
- **Augmentation:** Random flip, rotation, brightness, contrast, saturation, zoom — applied to training only
- **Class balancing:** Balanced oversampling across all six classes to address the underrepresented trash category (127 images)

### Model Training
All three models use **two-stage fine-tuning**:
- **Stage 1:** Frozen backbone — classification head trained only
- **Stage 2:** Selective unfreezing — 50% of transformer blocks (ViT) or 25% of convolutional layers (CNNs)

| Setting | ViT | CNNs |
|---|---|---|
| Optimizer | AdamW | Adam |
| Learning Rate | 2e-5 | 1e-4 |
| Weight Decay | 0.01 | 0.0001 |
| Batch Size | 16 | 32 |
| Early Stopping | patience=10 | patience=10 |

### Evaluation Protocol
- 5-Fold Stratified Cross-Validation for hyperparameter selection
- Macro F1 as primary metric (accounts for class imbalance)
- McNemar's test for statistical significance of performance differences

### Explainability (XAI)
- **ViT:** Attention map extraction from final encoder block (Block 11), averaged across 12 attention heads, upsampled to 224×224
- **CNNs:** Grad-CAM on last convolutional layer (`conv5_block3_out` for ResNet, `Conv_1` for MobileNet)

---

## Streamlit Application

A fully functional web application that allows users to:
- Upload any waste image
- Select from all three trained models
- Receive real-time classification with confidence score
- View the corresponding XAI visualisation (attention map or Grad-CAM)

### Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/SiyabongaDunn/WASTE-CLASSIFICATION-SYSTEM.git
cd WASTE-CLASSIFICATION-SYSTEM
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline (Google Colab recommended)

Open `notebooks/Full_Pipeline(Latest version).ipynb` in Google Colab — the dataset downloads automatically via `kagglehub`.

---

## Requirements

Key dependencies:

```
tensorflow>=2.13.0
transformers>=4.35.0
torch>=2.0.0
streamlit>=1.28.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
opencv-python>=4.8.0
statsmodels>=0.14.0
kagglehub
```

Full list available in `requirements.txt`.

---

## Reproducibility

All experiments are fully reproducible:
- Global random seed fixed at **42** across Python, NumPy, and TensorFlow
- Stratified splits ensure consistent class proportions across all subsets
- All hyperparameters documented in the pipeline notebook
- Library versions pinned in `requirements.txt`

---

## Key Findings

1. **ViT significantly outperforms CNNs** — 91.3% accuracy vs 84.8% (ResNet) and 83.8% (MobileNet) in baseline configuration, confirmed statistically at p < 0.001
2. **Class-balanced oversampling is the best strategy** — achieves 94.3% accuracy and 0.915 Macro F1
3. **ViT attention maps are more interpretable** — globally focused and coherent compared to CNN Grad-CAM which highlights local texture artefacts
4. **Both CNNs perform equivalently** — no statistically significant difference between ResNet50V2 and MobileNetV2 (p = 0.596), confirming the performance gain is architectural

---

## Future Work

- Model compression (quantization and pruning) for edge deployment
- Hardware validation on NVIDIA Jetson platforms
- In-facility testing at operational sorting facilities
- Extension to larger, more diverse waste datasets

---

## Citation

If you use this work, please cite:

```
Dunn, S. & Hlubi, A. (2026). Vision Transformer Architecture for Enhanced 
Global Feature Extraction in Automated Waste Sorting. Honours Dissertation, 
University of KwaZulu-Natal. Supervisor: Dr. A.A. Akinyelu.
```

---

## License

This project was developed as part of an Honours research dissertation at the University of KwaZulu-Natal. All code is available for academic and research purposes.
