import os
from datetime import datetime
import pickle
import pandas

def save_data(data, location, filename, method='pickle'):
    """
    
    Args:
        data - 
        location - 
        filename - 
        method - {'pickle', 'text'}
    """
    if method == 'pickle':
        file = open(f'{location}/{filename}', '')
        pickle.dump(data, file)

    elif method == 'text':
        file = open(f'{location}/{filename}', 'at')
        file.write(data)
    
    file.close()

    return None