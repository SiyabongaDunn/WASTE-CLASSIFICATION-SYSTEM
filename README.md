# WASTE-CLASSIFICATION-SYSTEM

A deep learning research project that compares Vision Transformer (ViT) and 
Convolutional Neural Network (CNN) architectures for automated waste image 
classification across six recyclable material categories.

**Authors:** Siyabonga Dunn (223017814) & Alunge Hlubi (223019985)  
**Supervisor:** Dr. Andronicus Ayobami Akinyelu  
**Institution:** University of KwaZulu-Natal  

---

## Project Overview

Global waste generation is projected to reach 3.4 billion tonnes by 2050. 
Current automated sorting systems rely on CNNs which process images locally 
— patch by patch — and struggle to distinguish visually similar materials 
such as clear glass and PET plastic. This project investigates whether Vision 
Transformers, which model global spatial relationships across an entire image 
simultaneously, can provide a more robust solution.

Three models are implemented and compared:

| Model | Type | Parameters |
|---|---|---|
| ViT-base-patch16-224 | Vision Transformer | ~86M |
| ResNet50V2 | CNN (deep residual) | ~25M |
| MobileNetV2 | CNN (lightweight) | ~3.4M |

---

## Dataset

**Kaggle Garbage Classification Dataset**  
2,527 images across six waste categories:

| Class | Images |
|---|---|
| Cardboard | 393 |
| Glass | 491 |
| Metal | 400 |
| Paper | 584 |
| Plastic | 472 |
| Trash | 127 |

Source: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
