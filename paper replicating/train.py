import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms

try:
    from torchinfo import summary
except:
    print("[INFO] couldn't find torchinfo!")


###loading the helper function
try:
    from going_modular import data_setup, engine
    from get_data import download_data
except:
    print("[INFO] couldn't load the helper functions!")

device = 'cuda' if torch.cuda.is_available() else 'mps'

print(f'[INFO] Using the device - {device}')

### getting the data
image_path = download_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
                           destination='pizza_steak_sushi')


train_dir = image_path / 'train'
test_dir = image_path /'test'


##image resolution as per the paper
IMG_SIZE = 224

## Create the transform pipeline accordingly 
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

print(f"Manual transforms are created -  {manual_transforms}")

###now geting the dataloaded 
BATCH_SIZE = 32

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir,
                                                                               transform=manual_transforms,
                                                                               batch_size=BATCH_SIZE)


###visualisng a single image
image_batch, label_batch = next(iter(train_dataloader))
# image, label = image_batch[0], label_batch[0]

# plt.imshow(image)
# plt.title(label)
# plt.axis(False)


