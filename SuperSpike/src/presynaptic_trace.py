import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

def presynaptic_trace(input_trains, args):
  """
  Evaluates the presynaptic trace (double exponential kernel)
  given the whole input trains
  Inputs:
    input_trains
    args:['timestep_size', 't_rise', 't_decay', 'nb_steps']
  Returns:
    Presynaptic Trace 
  """
  dt = args['timestep_size']
  t_rise = args['t_rise']
  t_decay = args['t_decay']
  nb_timesteps = args['nb_steps']
  device = args['device']
  dtype = args['dtype']
  
  nb_trains = len(input_trains)

  trace_1 = torch.zeros((nb_trains, nb_timesteps), device=device, dtype=dtype)
  trace_2 = torch.zeros((nb_trains, nb_timesteps), device=device, dtype=dtype)

  for t in range(nb_timesteps - 1):
    
  #  trace_1[:, t+1] = trace_1[:, t] + (-trace_1[:, t]/t_rise + input_trains[:, t])*dt
  #  trace_2[:, t+1] = trace_2[:, t] + (-trace_2[:, t] + trace_1[:, t])*dt/t_decay

    # New implementation (cognizant of the dirac delta):
    trace_1[:, t+1] = trace_1[:, t] + (-trace_1[:, t]/t_rise)*dt + input_trains[:, t]
    trace_2[:, t+1] = trace_2[:, t] + (-trace_2[:, t] + trace_1[:, t])*dt/t_decay
  
  return trace_2

def new_presynaptic_trace(input_trains, args):
  """
  Evaluates the presynaptic trace (double exponential kernel)
  given the whole input trains
  Inputs:
    input_trains
    args:['timestep_size', 't_rise', 't_decay', 'nb_steps']
  Returns:
    Presynaptic Trace, shape:(nb_inputs, nb_timesteps)
  """
  dt = args['timestep_size']
  t_rise = args['t_rise']
  t_decay = args['t_decay']
  nb_timesteps = args['nb_steps']
  device = args['device']
  dtype = args['dtype']
  nb_trains = len(input_trains)

  time_array = torch.arange(0, nb_timesteps*dt, dt, dtype=dtype, device=device)

  # create a model trace for a single spike at t=0, shape:(nb_timesteps)
  model_trace = (1/(t_rise-t_decay))*(torch.exp(-time_array/t_rise) - torch.exp(-time_array/t_decay))

  final_traces = torch.zeros((nb_trains, nb_timesteps), device=device, dtype=dtype)

  for t in range(nb_timesteps):
    indices = input_trains[:, t] == 1
    final_traces[indices, t:] += model_trace[:nb_timesteps - t]
  
  return final_traces