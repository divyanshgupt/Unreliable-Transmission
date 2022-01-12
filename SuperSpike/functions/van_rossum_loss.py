import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

def van_rossum_loss(output, target, args):
    """
    Evaluates the van rossum loss over the normalized double-exponential kernel
    Inputs:
        output - spiketrain, shape: (nb_steps,)
        target - spiketrain, shape: (nb_steps,)
    Returns
        van Rossum loss (over normalized double exponential error signal)
    """
    
    t_rise = args['t_rise_alpha']
    t_decay = args['t_decay_alpha']
    dt = args['timestep_size']
    nb_steps = args['nb_steps']
    device = args['device']
    dtype = args['dtype']

    difference = target - output
    # print(difference.shape)

    trace_1 = torch.zeros(nb_steps, device=device, dtype=dtype)
    trace_2 = torch.zeros(nb_steps, device=device, dtype=dtype)

    for t in range(nb_steps - 1):
        trace_1[t+1] = trace_1[t] + (-trace_1[t]/t_rise + difference[t])*dt
        trace_2[t+1] = trace_2[t] + (-trace_2[t]/t_decay + trace_1[t])*dt/t_decay

    # normalizing the double exponential convolution
    normalization_factor = torch.max(trace_2)
    # print("Normalization factor:", normalization_factor)
    trace_2 = trace_2/normalization_factor
    # print(torch.max(trace_2))
    loss = torch.sum(trace_2**2)*dt
    
    return loss