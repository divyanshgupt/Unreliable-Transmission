import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

import functions

def eligibility_trace(hebbian, args):
  """
  Evaluate the hebbian-coincidence based eligibility trace over all timesteps
  for all the given synaptic connections in the hebbian matrix using the 
  double exponential kernel.
  Inputs:
    hebbian - 2-D matrix of shape: (nb_inputs, nb_outputs)
    args: ['timestep_size', 't_rise_alpha', 't_decay_alpha', 'nb_steps']
  Returns:
    Eligibilty trace matrix of shape: (nb_inputs, nb_outputs, nb_timesteps)
  """
  dt = args['timestep_size']
  t_rise = args['t_rise_alpha']
  t_decay = args['t_decay_alpha']
  nb_timesteps = args['nb_steps']
  device = args['device']
  dtype = args['dtype']
  
  nb_inputs = hebbian.shape[0]
  nb_outputs = hebbian.shape[1]

  trace_1 = torch.zeros((nb_inputs, nb_outputs, nb_timesteps), device=device,
                       dtype=dtype)
  trace_2 = torch.zeros((nb_inputs, nb_outputs, nb_timesteps), device=device,
                        dtype=dtype)

  for t in range(nb_timesteps-1):
    trace_1[:, :, t+1] = trace_1[:, :, t] + (-trace_1[:, :, t]/t_rise + hebbian[:, :, t])*dt
    trace_2[:, :, t+1] = trace_2[:, :, t] + (-trace_2[:, :, t] + trace_1[:, :, t])*dt/t_decay

  return trace_2

def new_eligibility_trace(hebbian, args):
  """
  Evaluate the hebbian-coincidence based eligibility trace over all timesteps
  for all the given synaptic connections in the hebbian matrix using the 
  double exponential kernel, using the Runge-Kutta method
  Inputs:
    hebbian - 2-D matrix of shape: (nb_inputs, nb_outputs)
    args: ['timestep_size', 't_rise_alpha', 't_decay_alpha', 'nb_steps']
  Returns:
    Eligibilty trace matrix of shape: (nb_inputs, nb_outputs, nb_timesteps)
  """
  
  dt = args['timestep_size']
  t_rise = args['t_rise_alpha']
  t_decay = args['t_decay_alpha']
  nb_timesteps = args['nb_steps']
  device = args['device']
  dtype = args['dtype']
  
  nb_inputs = hebbian.shape[0] # inferring from hebbian shape is to ensure that the shape(input x output) generalizes for multilayer network
  nb_outputs = hebbian.shape[1]

  trace_1 = torch.zeros((nb_inputs, nb_outputs, nb_timesteps), device=device,
                       dtype=dtype)
  trace_2 = torch.zeros((nb_inputs, nb_outputs, nb_timesteps), device=device,
                        dtype=dtype)

  f = lambda x, hebbian: -(x/t_rise) + hebbian # takes x and hebbian at a time step, shape:(nb_inputs, nb_outputs)
  g = lambda x, trace_1: (-x + trace_1)/t_decay
  
  for t in range(nb_timesteps-1):
    k1 = f(trace_1[:, :, t], hebbian[:, :, t])*dt
    k2 = f(trace_1[:, :, t] + (k1/2), hebbian[:, :, t])*dt
    k3 = f(trace_1[:, :, t] + (k2/2), hebbian[:, :, t])*dt
    k4 = f(trace_1[:, :, t] + k3, hebbian[:, :, t])*dt

    trace_1[:, :, t+1] = trace_1[:, :, t] + (1/6)*(k1 + (2*k2) + (2*k3) + k4)

    p1 = g(trace_2[:, :, t], trace_1[:, :, t])*dt
    p2 = g(trace_2[:, :, t] + (p1/2), trace_1[:, :, t])*dt
    p3 = g(trace_2[:, :, t] + (p2/2), trace_1[:, :, t])*dt
    p4 = g(trace_2[:, :, t] + p3, trace_1[:, :, t])*dt

    trace_2[:, :, t+1] = trace_2[:, :, t] + (1/6)*(p1 + (2*p2) + (2*p3) + p4)

  return trace_2