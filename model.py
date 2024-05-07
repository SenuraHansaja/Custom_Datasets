import torch
import os
import requests
import zipfile
from pathlib import Path
from torch.utils.data import dataloader
from torchvision import datasets, transforms

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

def walk_through(dir_path):
    '''
    arg  : get the directory path 
    return :  the content of the directory 
    
    '''
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f'The are {len(dirnames)} directories and {len(filenames)} in the {dirpath}')
        
##setting up the train and test path 

train_dir = image_path / "train"
test_dir = image_path / "test"



### data augmentation scheme
data_transform = transforms.Compose([
    #resize the image
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor()
])