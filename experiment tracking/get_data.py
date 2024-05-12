import os
from pathlib import Path
import zipfile
import requests

def download_data(
    source:str, 
    destination: str, 
    remove_source:bool =  True
)-> Path:
    ''' Download a zip dataset and unzip to a destination datafolder'''
    '''
    Args : 
        source : the string to get the data from 
        destination: the desitnation folder to save the unzip data to 
        remove_source : whehter to remove or keep the source folder after creaing
        
    returns
        returns: path lib to download the data
    '''
    data_path = Path("data/")
    image_path = data_path / destination

    if image_path.is_dir():
        print(f'[INFO]{image_path} directory already exist....')
    
    else:
        print(f'[INFO] image directory creating.....')
        image_path.mkdir(parents= True, exist_ok=True)

        ###downloading the data
        target_file = Path(source).name
        with open(data_path /  target_file, "wb") as f:
            request = requests.get(source)
            print(f'[INFO] downloading the data to the {target_file} from the {source}.....')
            f.write(request.content)

        ##unzipping the data
        with zipfile.ZipFile(data_path / target_file, 'r') as zip_ref:
            print(f"[INFO] Unzipping the data {target_file}  data ....")
            zip_ref.extractall(image_path)

    if remove_source == True:
        os.remove(data_path / target_file)

    return image_path


image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")

print(image_path)