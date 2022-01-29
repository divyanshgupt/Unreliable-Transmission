import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

import functions

def run_single_neuron(input_trains, weights, target, args):
  """
  Run a single LIF neuron as specificed by Zenke Ganguli for one epoch
  Inputs:
    input_trains
    weights
    target - desired spike train
    args['thres']
    args['u_rest']
  Returns:
    mem_rec, spk_rec, eligibility_rec, presynaptic_traces, error_signal

  """
  thres = args['thres']
  u_rest = args['u_rest']
  nb_steps = args['nb_steps']
  device = args['device']
  dtype = args['dtype']
  nb_inputs = args['nb_inputs']
  nb_outputs = args['nb_outputs']
  alpha = args['alpha']
  beta = args['beta']

  # initialize membrane and synaptic current 
  mem = u_rest * torch.ones(nb_outputs, device=device, dtype=dtype)
  syn = torch.zeros(nb_outputs, device=device, dtype=dtype)

  # lists to store traces and neuron variables
  mem_rec = []
  spk_rec = []
  eligibility_rec = []
  pre_trace_rec = []
  error_rec = []

  reset = 0

  for t in range(nb_steps):

    out = functions.spike_fn(mem, thres, args)

    spk_rec.append(out)
    mem_rec.append(mem)

    # refractory period = 5 ms = 50 timesteps of 0.1 ms each
    if t < 50:
      if 1 in spk_rec:
        reset = 1
     #   print("Reseting membrane potential to resting value")
    elif 1 in spk_rec[-50:]:
      reset = 1
    else:
      reset = 0

    # LIF membrane potential update
    weighted_inp = torch.sum(input_trains[:, t] * weights.T)
    new_syn = alpha*syn + weighted_inp
    new_mem = (beta*mem + syn*(1 - beta) + u_rest*(1 - beta))*(1 - reset) + (reset * u_rest)

    mem = new_mem
    syn = new_syn
  # end of simulation loop
  
  mem_rec = torch.stack(mem_rec, dim=0)
  spk_rec = torch.stack(spk_rec, dim=0)
  spk_rec = torch.flatten(spk_rec) # stack as a 1-D array for easy difference with target train for error signal evaluation
  # compute presynaptic traces of shape: (nb_inputs, timesteps) 
  presynaptic_traces = functions.new_presynaptic_trace(input_trains, args) 

  # evaluate hebbian coincidence
  h = mem_rec.T - thres  # shape: (nb_outputs, timesteps)
  post = 1 / (1 + torch.abs(h))**2 # shape: (nb_outputs, timesteps) 
  A = torch.unsqueeze(presynaptic_traces, 1)
  B = torch.unsqueeze(post, 0)
  hebbian = A * B # AB.T shape: (nb_inputs, nb_outputs, nb_timesteps)

  # eligibility trace
  eligibility = functions.eligibility_trace(hebbian, args)
  
  # error signal
  error = functions.new_error_signal(spk_rec, target, args)

  return mem_rec, spk_rec, error, eligibility, presynaptic_traces