import matplotlib.pyplot as plt 
import torch
import torchvision

from torch import nn
from torchvision import transforms

###hyper parameters
BATCH_SIZE = 32
NUM_WORKER = 1



# ## try to get the torch info installed to the code
try :
    import torchinfo
except:
    print("Couldn't find the specific package...")


## importing the previous data and function calls
try :
    import train
    import engine
    from get_data import download_data
    from data_steup import create_dataloaders

except:
    print("couldn't load the dependencies")
    

device = "cuda" if torch.cuda.is_available() else "cpu"

if device != 'cuda':
    device = "mps"
print(f'[INFO] the model use - {device}')

##setting up directories
from get_data import download_data
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")
train_dir = image_path  / "train"
test_dir = image_path / "test"

normalize = transforms.Normalize(mean =[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) ##noramlize to image net data

manual_tranforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    normalize    
])

print(f'Manually created transforms: {manual_tranforms}')

### create the data loaders for train and test

train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    transforms=manual_tranforms,
                                                                    BATCH_SIZE = BATCH_SIZE,
                                                                    NUM_WORKER = NUM_WORKER)




