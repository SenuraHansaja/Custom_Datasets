import torch
import os
import requests
import zipfile
import torchvision
from pathlib import Path
from torchvision import transforms
from data_setup import create_dataloaders

device = 'mps'

data_path = Path('data/')
image_path = data_path / "image path"

train_dir = image_path / "train"
test_dir = image_path / "test"

manual_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

train_dataloader, test_dataloader, class_name = create_dataloaders(train_dir=train_dir, 
                                                                   test_dir=test_dir,
                                                                   transforms=manual_transform,
                                                                   batch_size=32)



##what below code line does is get the most updated weights of the model by default
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

auto_transform = weights.transforms()

rain_dataloader, test_dataloader, class_name = create_dataloaders(train_dir=train_dir, 
                                                                   test_dir=test_dir,
                                                                   transforms=auto_transform,
                                                                   batch_size=32)

model = torchvision.models.efficientnet_b0(weights=weights).to(device)  ##this is the standard new method of using the pretrained model to load the model



