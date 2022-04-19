import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

def error_signal(output, target, args):
  """
  Evaluates the error signal by running the double exponential 
  kernel on the difference of the output and the target spike trains.
  Inputs:
    output - spike_train, shape: (nb_steps,)
    target - spike_train, shape: (nb_steps,)
    args:['timestep_size', 't_rise_alpha', 't_decay_alpha', 'nb_steps']
  Returns
    Error Signal Trace of shape: (nb_steps,)
  """
  t_rise = args['t_rise_alpha']
  t_decay = args['t_decay_alpha']
  dt = args['timestep_size']
  nb_steps = args['nb_steps']
  device = args['device']
  dtype = args['dtype']

  trace_1 = torch.zeros(nb_steps, device=device, dtype=dtype)
  trace_2 = torch.zeros(nb_steps, device=device, dtype=dtype)

 # print("Target Shape", target.shape)
 # print("Output Shape", output.shape)
  difference = target - output
 # print("Difference Shape", difference.shape)

  for t in range(nb_steps - 1):
  #  trace_1[t + 1] = trace_1[t] + (-trace_1[t]/t_rise + difference[t])*dt
  #  trace_2[t + 1] = trace_2[t] + (-trace_2[t] + trace_1[t])*dt/t_decay

  # New implementation (cognizant of the dirac delta):
    trace_1[t+1] = trace_1[t] + (-trace_1[t]/t_rise)*dt + difference[t]
    trace_2[t+1] = trace_2[t] + (-trace_2[t] + trace_1[t])*dt/t_decay

  return trace_2


def new_error_signal(output, target, args):
  """
  Evaluates the error signal by running the double exponential 
  kernel on the difference of the output and the target spike trains.
  Inputs:
  output - spike_train, shape: (nb_steps,)
  target - spike_train, shape: (nb_steps,)
  args:['timestep_size', 't_rise_alpha', 't_decay_alpha', 'nb_steps']
  Returns
  Error Signal Trace of shape: (nb_steps,)
  """
  t_rise = args['t_rise_alpha']
  t_decay = args['t_decay_alpha']
  dt = args['timestep_size']
  nb_timesteps = args['nb_steps']
  device = args['device']
  dtype = args['dtype']
  nb_inputs = args['nb_inputs']
  nb_outputs = args['nb_outputs']
  
  difference = target - output # can take values:{-1, 0, 1}

  time_array = torch.arange(0, nb_timesteps*dt, dt, dtype=dtype, device=device)

  # create a model trace for a single spike at t=s0, shape:(nb_timesteps)
  model_trace = (1/(t_rise-t_decay))*(torch.exp(-time_array/t_rise) - torch.exp(-time_array/t_decay))

  final_trace = torch.zeros(nb_timesteps, device=device, dtype=dtype)

  for t in range(nb_timesteps):
 
    if difference[t] == 1:
      final_trace[t:] += model_trace[:nb_timesteps - t]
    elif difference[t] == -1:
      final_trace[t:] -= model_trace[:nb_timesteps - t]
  
  return final_trace