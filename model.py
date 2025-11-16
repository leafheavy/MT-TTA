
import torch
import torch.nn as nn
from utils import *
from torchvision import models

from utils import model_path, plantdata_task_classes, device

import os
os.environ['TORCH_HOME'] = os.path.join(model_path, 'cnn')   

class MyModel(nn.Module):
    def __init__(self, task_classes, resnet='resnet18', device=device):
        """
        - args:
            - task_classes: A list of the number of classes for each task. 
                - For example: [2, 2, 2] indicates three tasks, and each task has two classes
            - resnet: The selected ResNet models ('resnet18', 'resnet50', 'resnet101')
        """
        super().__init__()
        self.device = device
        if resnet == 'resnet18':
            self.backbone = models.resnet18(weights=None).to(device)
            self.backbone.fc = nn.Identity()
        elif resnet == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
            self.backbone.fc = nn.Identity()
        elif resnet == 'resnet101':
            self.backbone = models.resnet101(weights=None).to(device)
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported ResNet model: {resnet}")

        # Constructing multi-task classifiers
        self.task_heads = nn.ModuleList()
        for num_classes in task_classes:
            if resnet == 'resnet18':
                head = nn.Linear(512, num_classes)
            elif resnet == 'resnet50':
                head = nn.Linear(2048, num_classes)
            elif resnet == 'resnet101':
                head = nn.Linear(2048, num_classes)
            self.task_heads.append(head)
        
        self.to(device)

    def forward(self, x):
        x = x.to(torch.float32)
        
        features = self.backbone(x)
        
        outputs = [head(features) for head in self.task_heads]
        return outputs
