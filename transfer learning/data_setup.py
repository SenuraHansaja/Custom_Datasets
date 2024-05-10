import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


NUM_WORKERS = os.cpu_count()

def create_dataloaders (
        train_dir :str,
        test_dir : str,
        transforms : transforms.Compose,
        batch_size : int,
        num_workers : int = NUM_WORKERS
):
    """create traning and testing dataloader"""
    train_data =  datasets.ImageFolder(train_dir, transforms=transforms)
    test_data = datasets.ImageFoldera(test_dir, transforms=transforms)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data, 
        batch_size = batch_size,
        shuffle = True, 
        num_workers = num_workers, 
        pin_memory = True
    )

    test_dataloader = DataLoader(
        test_data, 
        batch_size = batch_size, 
        shuffle = False, 
        num_workers = num_workers, 
        pin_memory  = True
    )

    return train_dataloader, test_dataloader, class_names