import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import functions

def heterosynaptic_regularization(activity, args):
    """
    Evaluates the heterosynaptic regularization term by computing the 
    exponential moving average of the instantaeous post-synaptic firing rate
    
    Returns:
        regularization - shape:(nb_)
    """

    device = args['device']
    dtype = args['dtype']
    nb_steps = args['nb_steps']
    dt = args['timestep_size']
    tau_het = args['tau_het']
    a = args['a'] # exponent parameter for regularization evalution

    nb_outputs = activity.shape[0]

    regularization = torch.zeros((nb_outputs, nb_steps), device=device, dtype=dtype)

    time_array = torch.arange(0, nb_steps*dt, dt, device=device, dtype=dtype)
    model_trace = torch.exp(-time_array/tau_het).to(device)

    for t in range(nb_steps):
        indices = activity[:, t] >= 1
        regularization[indices, t:] += model_trace[:nb_steps - t]

    regularization = torch.sum(regularization**a, dim=1)*dt # simple first order Euler integration method

    assert regularization.shape == (nb_outputs,), "Regularization term shape incorrect"

    return regularization

