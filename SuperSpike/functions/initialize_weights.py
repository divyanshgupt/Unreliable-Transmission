import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

def initialize_weights(nb_inputs, nb_outputs, args, scale=70):
    """
    Inputs:
        nb_inputs - 
        nb_outputs - 
        scale - 
    Returns:
        weights: shape:(nb_inputs, nb_outputs)
    """
    device = args['device']
    dtype = args['dtype']
    
    weight_scale = scale*(1 - 0) # copied from spytorch

    weights = torch.empty((nb_inputs, nb_outputs), device=device, dtype=dtype)
    weights = torch.nn.init.normal_(weights, mean=0.5, std=weight_scale/np.sqrt(nb_inputs))
    print("Weight initialization done")
    return weights

def initialize_weights_multilayer(nb_inputs, nb_hidden, nb_outputs, args, weight_scale=70):
    """
    Inputs:
        nb_inputs
        nb_hidden
        nb_outputs
        args
        weight_scale

    Returns:
        w1 - shape:(nb_inputs, nb_hidden)
        w2 - shape:(nb_hidden, nb_outputs)
    """

    device = args['device']
    dtype = args['dtype']

    weight_scale = weight_scale*(1 - 0)
    # Weights from Input Layer to Hidden Layer
    w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype)
    w1 = torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))
    # Weights from Hidden Layer to Output Layer
    w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype)
    w2 = torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

    return w1, w2