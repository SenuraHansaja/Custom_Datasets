import torch
import requests
from pathlib import Path
import zipfile

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print('f {image_path} directory exist')

else:
    print(f"did not find the path {image_path} creating a new one....")
    image_path.mkdir(parents=True, exist_ok=False)


with open(data_path/'pizza_sushi_steak.zip', 'wb') as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    f.write(request)

with zipfile.ZipExtFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza steak, sushi data")
    zip_ref.extractall(image_path)