import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import yaml
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
import torch.cuda.amp as amp

def set_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FridgeNet(nn.Module):
    def __init__(self, reg_attrs: List[str], cls_attrs: Dict[str, int], num_classes: int = 6, mode = 'both'):

        backbone = models.resnet18(backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1))
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if mode in ['both', 'attr']:
            se

        elif mode in ['both', 'type']:
            self.fc = nn.Linear(in_feat, num_classes)
            
            def forward_type()