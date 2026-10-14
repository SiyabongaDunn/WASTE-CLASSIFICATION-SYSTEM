# Vision Transformer Architecture for Enhanced Global Feature Extraction in Automated Waste Sorting and Sustainability

## Authors

| Name | Email |
|---|---|
| Siyabonga Dunn | [223017814@stu.ukzn.ac.za](mailto:223017814@stu.ukzn.ac.za) |
| Alunge Hlubi | [223019985@stu.ukzn.ac.za](mailto:223019985@stu.ukzn.ac.za) |

---

## Overview

The increasing volume and complexity of solid waste present significant challenges for efficient recycling and automated waste-sorting systems. Computer vision is now a core component of automated sorting, with convolutional neural networks (CNNs) widely applied to waste image classification. However, the local receptive-field structure of CNNs can limit their ability to capture global spatial relationships in images containing overlapping, occluded, or visually ambiguous waste items.

Vision Transformers (ViTs) offer an alternative: self-attention models relationships between image regions globally. This project investigates the effectiveness of a pretrained **ViT-B/16** for automated waste classification on the **TrashNet** dataset and compares it against **ResNet50V2** and **MobileNetV2** CNN baselines. It also examines how **targeted augmentation** and **class-balanced oversampling** affect the underrepresented *Trash* category.

The best-performing model, an oversampled ViT, is deployed in a Streamlit web application that returns a prediction, a confidence score, and an attention heatmap.

---

## Key Results

| ViT-B/16 Configuration | Test Accuracy | Macro F1 |
|---|---|---|
| Baseline (two-stage fine-tuning, augmentation, class weights) | 91.3% | 0.885 |
| Targeted augmentation (Trash class only) | 89.7% | 0.874 |
| **Class-balanced oversampling (best)** | **94.3%** | **0.915** |

- Class-balanced oversampling gave the strongest overall result.
- Targeted augmentation did not improve on the baseline.
- Per-class F1 scores, confusion matrices, McNemar's test results, and the ResNet50V2 / MobileNetV2 comparisons are reported in the full paper.

---

## Research Questions

1. Can a transfer-learned ViT classify waste effectively on a small dataset?
2. How does its performance compare with ResNet50V2 and MobileNetV2?
3. Do targeted augmentation and class-balanced oversampling improve results on the underrepresented *Trash* class?
4. Are model decisions interpretable, and do they rely on relevant object features rather than background noise?

---

## Methodology

| Component | Details |
|---|---|
| **Dataset** | TrashNet, 6 classes (cardboard, glass, metal, paper, plastic, trash), 2,527 images |
| **Data split** | 80% training pool, 20% untouched test set (506 images) |
| **Validation** | Stratified 5-fold cross-validation on the training pool |
| **Models** | ViT-B/16 (pretrained, transfer learning), ResNet50V2, MobileNetV2 |
| **Imbalance strategies** | Targeted augmentation, class-balanced oversampling |
| **Metrics** | Accuracy, precision, recall, macro F1, per-class F1, confusion matrices |
| **Statistical testing** | McNemar's test |
| **Explainability** | ViT attention visualization; Grad-CAM for CNNs |

---

## Explainability

| Model | Method | What it shows |
|---|---|---|
| ViT-B/16 | Attention visualization | Image regions the self-attention layers focus on |
| ResNet50V2, MobileNetV2 | Grad-CAM | Regions that drive each CNN prediction |

Explainability is model-specific: each architecture needs a method suited to how it represents information.

---

## Web Application Prototype

The oversampled ViT is integrated into a Streamlit application to demonstrate real-world deployment viability. For each uploaded image, the app displays:

- The predicted waste class
- A confidence score
- An attention heatmap showing which regions influenced the prediction

**Example:** When tested with a challenging image of a dirty, folded tissue, the model classified it as **Trash with 99.7% confidence** (Fig. 10 in the paper). The heatmap localized the object's irregular textures, showing the model relied on relevant structural features rather than background noise, which makes for a functional and transparent workflow.

---

## Tech Stack

- **Language:** Python
- **Deep learning:** TensorFlow 2.20 / Keras 3.13 (native Keras ViT-B/16 implementation with pretrained weights transferred from HuggingFace)
- **Training environment:** Google Colab
- **Deployment:** Streamlit

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/SiyabongaDunn/WASTE-CLASSIFICATION-SYSTEM.git
cd WASTE-CLASSIFICATION-SYSTEM

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`), upload a waste image, and view the prediction and attention heatmap.

---

## Dataset

This project uses the [TrashNet](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) dataset. The dataset is not redistributed here; download it from the original source if you wish to retrain or re-evaluate the models.

---

## Conclusions

- Transfer-learned ViTs show strong potential for small waste-classification datasets.
- Data-balancing strategies matter: class-balanced oversampling clearly outperformed targeted augmentation.
- Model-specific explainability supports transparent, trustworthy deployment.

---

## Citation

If you use this work, please cite:

```bibtex
@unpublished{dunn2026vitwaste,
  author      = {Dunn, Siyabonga and Hlubi, Alunge},
  title       = {Vision Transformer Architecture for Enhanced Global Feature Extraction in Automated Waste Sorting and Sustainability},
  institution = {University of KwaZulu-Natal},
  year        = {2026}
}
```

---

## License

Released under the **MIT License**, free for academic use. See [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

We thank our supervisor, **Andronicus Ayobami Akinyelu**, for his guidance, the Discipline of Computer Science at the University of KwaZulu-Natal, and the creators of the TrashNet dataset.
