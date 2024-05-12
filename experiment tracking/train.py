import matplotlib.pyplot as plt 
import torch
import torch.utils
import torch.utils.data
import torchvision

from torch import nn
from torchvision import transforms

from tqdm.auto import tqdm

###hyper parameters
BATCH_SIZE = 32
NUM_WORKER = 1



# ## try to get the torch info installed to the code
try :
    import torchinfo
except:
    print("Couldn't find the specific package...")


## importing the previous data and function calls
import engine
from get_data import download_data
from data_steup import create_dataloaders

    

device = "cuda" 
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
    nn.Linear(in_features = 1280, out_features = len(class_names), bias = True).to(device)
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

from typing import Dict, List
from tqdm.auto import tqdm

from engine import train_step, test_step

def train (model : torch.nn.Module,
           train_dataloader : torch.utils.data.DataLoader,
           test_dataloader: torch.utils.data.DataLoader,
           optimizer : torch.optim.Optimizer,
           loss_fn : torch.nn.Module,
           epochs  : int,
           device : torch.device) -> Dict[str, list]:
    
    """ takes the model and the daataloader and test dataloader , and optimizer and the loss function nd the number of epochs along with the deivce
    and then return the dictionary containing the train loss and test loss and trai acc and test acc"""
    
    results  = {'train_loss': [],
                'test_loss':  [],
                'train_acc' : [],
                'test_acc' : []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer,
                                           devise=device)
        
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
        
        
        ## updating the resutls
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        print(f"Epoch {epoch} Train Loss - {train_loss} : Train acc - {train_acc} : Test loss {test_loss} test acc {test_acc}")
        ## adding the accuracy results to the summary writer
        # writer.add_scalars(main_tag ="Accuracy", 
        #                   tag_scalar_dict = {"train_acc": train_acc,
        #                                      "test_acc": test_acc},
        #                   global_step=epoch)
        # ### track the pytorch model architecture
        # writer.add_graph(model=model, 
        #                  ## pass in an exampple input to be tracked
        #                  input_to_model=torch.randn(32, 3, 224, 224))
        # writer.close()
        
        return results
    
    
    
# results = train(model.to('cuda'), 
#                 train_dataloader=train_dataloader,
#                 test_dataloader=test_dataloader,
#                 optimizer=optimizer,
#                 loss_fn=loss_fn,
#                 epochs=10,
#                 device='cuda')

# print(results['test_acc'])
##initialising the wandb

import wandb

wandb.init(
    project='experiment tracking learnpytorch.io',
    config= {'learning_rate':0.001,
             'architecture': "efficientnet_b0",
             'dataset' : 'subset of food101',
             'epochs' : 5
             }
)
food_101_artifact = wandb.Artifact("food101", type="dataset")
file_path = '/home/ifire/learnpytorh/Custom_Datasets/experiment tracking/data/pizza_steak_sushi'
food_101_artifact.add_dir(file_path)




EPOCHS = 5
results  = {'train_loss': [],
                'test_loss':  [],
                'train_acc' : [],
                'test_acc' : []}

wandb.watch(model, loss_fn, log= "all", log_freq=1)
for epoch in range(EPOCHS):
    train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer,
                                           devise=device)
        
    test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
    
    
    
    wandb.log({'train loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc})

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
    print(f"Epoch {epoch} Train Loss - {train_loss} : Train acc - {train_acc} : Test loss {test_loss} test acc {test_acc}")

