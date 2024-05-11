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