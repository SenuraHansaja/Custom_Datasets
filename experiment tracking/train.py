import matplotlib.pyplot as plt 
import torch
import torchvision

from torch import nn
from torchvision import transforms

# ## try to get the torch info installed to the code
try :
    import torchinfo
except:
    print("Couldn't find the specific package...")


## importing the previous data and function calls
try :
    import train
    import engine

except:
    print("couldn't load the dependencies")
    

device = "cuda" if torch.cuda.is_available() else "cpu"

if device != 'cuda':
    device = "mps"

print(f'[INFO] the model use - {device}')