import torch
from transformers import ViTModel
import numpy as np

_pt_weights = None

def load_pretrained_weights(model_name='google/vit-base-patch16-224'):
    global _pt_weights
    
    if _pt_weights is not None:
        print(" Using cached pretrained weights")
        return _pt_weights
    
    print(f"Loading pretrained weights from HuggingFace: {model_name}")
    pt_model = ViTModel.from_pretrained(model_name)
    pt_model.eval()
    
    _pt_weights = {
        name: param.detach().numpy() 
        for name, param in pt_model.named_parameters()
    }
    
    print(f" Loaded {len(_pt_weights)} weight tensors")
    return _pt_weights


def get_weight(key):
    weights = load_pretrained_weights()
    return weights[key]