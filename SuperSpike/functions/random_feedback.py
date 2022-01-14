
import torch
import numpy as np

def random_feedback(nb_hidden, nb_outputs, args, weight_scale):
    """
    Generates random feedback weights based on given layer sizes
    Inputs:
        nb_hidden - 
        nb_outputs - 
        weight_scale - the std deviation is weight_scale/np.sqrt(nb_inputs)
    Returns:
        Weight matrix of shape: (nb_inputs, nb_outputs)
    """

    device = args['device']
    dtype = args['dtype']
    nb_inputs = args['nb_inputs']
    
    b = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype)
    b = torch.nn.init.normal_(b, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

    return b