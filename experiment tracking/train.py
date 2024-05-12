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
import train
import engine
from get_data import download_data
from data_steup import create_dataloaders

    

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
                                                                    batch_size = BATCH_SIZE,
                                                                    num_workers = NUM_WORKER)


## now we will use the automati tranformations

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
automatic_transformation = weights.transforms()
print(f'automatic transformation are listed as here {automatic_transformation}')

##now using the auotmatic transformation we will load the dataloaders

train_dataloader, test_dataloader, classes = create_dataloaders(train_dir=train_dir,
                                                                test_dir=test_dir,
                                                                transforms=automatic_transformation,
                                                                batch_size= BATCH_SIZE,
                                                                num_workers=NUM_WORKER)

##downloading the efficient net pretrained model

model = torchvision.models.efficientnet_b0(weights=weights).to(device)

## changing the output head of the mdoel to match with the dataset
for param in model.parameters():
    param.requires_grad = False

##seting the classifier to match with our problem 
model.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features = 12800, out_features = len(class_names), bias = True).to(device)
)
## to print a model summary 
from torchinfo import summary

# # Get a summary of the model (uncomment for full output)
# summary(model, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape" (batch_size, color_channels, height, width)
#         verbose=0,
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


## to track the experiments results and visualise them

from torch.utils.tensorboard import SummaryWriter

# Create a writer with all default settings
writer = SummaryWriter()