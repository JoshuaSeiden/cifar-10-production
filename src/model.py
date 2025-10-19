import torch
import torch.nn as nn
import torchvision.models as models
import os
os.environ['TORCH_HOME'] = '/tmp/torch'  # or './cache' for relative to repo

def get_resnet50_model(num_classes=10):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
