import os
from datetime import datetime
import pickle
import pandas


def set_location(folder_name):
    """
    Returns complete path of the folder name and creates it if doesn't exist
    Args:
        folder_name
    Returns:
        location (string)
    """
    
    location = os.path.abspath(folder_name)
    location = os.path.join(os.getcwd(), location)
    if not os.path.isdir(location):
        os.makedirs(location)

    return location


def save_data(data, location, filename, method='pickle'):
    """
    Save the given data into the given location with the given filename and the method.

    Args:
        data - 
        location - 
        filename - 
        method - {'pickle', 'text'}

    Returns:
        None
    """
    if not method in ['pickle', 'text']:
        raise NotImplementedError("Saving method not specified properly")

    if method == 'pickle':
        file = open(f'{location}/{filename}', '')
        pickle.dump(data, file)

    elif method == 'text':
        file = open(f'{location}/{filename}', 'at')
        file.write(data)
    
    file.close()

    return None