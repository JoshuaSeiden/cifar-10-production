import torch
import torch.nn as nn
import torchvision.models as models
import os
os.environ['TORCH_HOME'] = '/tmp/torch'

def get_resnet18_model(num_classes=10):
    model = models.resnet18(weights=None)
    for p in model.parameters():
        p.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
