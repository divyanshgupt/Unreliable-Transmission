# Step Function for Spikes
import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

def spike_fn(x, thres, args):
  """
  Implements a heaviside function centred at the firing threshold
  """

  device = args['device']
  dtype = args['dtype']
  
  x = x - thres
  out = torch.zeros_like(x, device=device, dtype=dtype)
  out[x > 0] = 1
  return out