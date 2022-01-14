import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch


import functions

# set device
dtype = torch.float
cpu = torch.device("cpu")
gpu = torch.device("cuda:0")

device = cpu
# Uncomment the line below to run on GPU
#device = gpu

args = {'thres': -50,
        'u_rest': -60,
        'tau_mem': 1e-2,
        'tau_syn': 5e-3,
        'tau_ref': 5e-3,
        't_rise': 5e-3, # the pre-synaptic double exponential kernel rise time
        't_decay': 1e-2, # the pre-synaptic double exponential kernel decay time
        'timestep_size': 1e-4, # 0.1 msec
        't_rise_alpha': 5e-3, # change this
        't_decay_alpha': 1e-2, # change this 
        'nb_steps': 5000, # 0.5 secs in total
        'tau_rms': 5e-3,
        'nb_inputs': 100,
        'nb_hidden': 4,
        'nb_outputs': 1,

        } 
    
nb_inputs = args['nb_inputs']
nb_hidden = args['nb_hidden']
nb_outputs = args['nb_outputs']

nb_steps = args['nb_steps']
dt = args['timestep_size']

# 100 independent poisson trains
input_trains = functions.poisson_trains(nb_inputs, 10*np.ones(100), nb_steps, dt)

# 5 equidistant spikes over 500 msec.
target = torch.zeros((nb_steps), device=device, dtype=dtype)
target[::int(nb_steps/5)] = 1

#@title Training the network

epochs = 5
weight_scale = 100

w1, w2 = functions.initialize_weights_multilayer(nb_inputs, nb_hidden, nb_outputs, args, weight_scale)

feedback_weights = functions.random_feedback(nb_hidden, nb_outputs, weight_scale).T # shape: (nb_outputs, nb_hidden)
loss_rec = []

v_ij_1 = torch.ones((nb_inputs, nb_hidden), device=device, dtype=dtype)
v_ij_2 = torch.ones((nb_hidden, nb_outputs), device=device, dtype=dtype)

initial_rate = 5e-3
gamma = float(np.exp(-dt/args['tau_rms']))


for i in tqdm(range(epochs)):
  print("Epoch no:", i)
  eligibility_1, eligibility_2, presynaptic_trace_1, presynaptic_traces_2, spk_rec_2, spk_rec_1, mem_rec_2, mem_rec_1 = functions.run_multilayer_network(input_trains, w1, w2, args)
  
  output = torch.flatten(spk_rec_2)

  # Evaluate Loss & Store for Plotting:
  loss = functions.van_rossum_loss(target, output, args)
  loss_rec.append(loss)
  print("Loss =", loss)

  # Evaluate Error Signal for both layers:
  output_error = torch.unsqueeze(functions.error_signal(output, target, args), 1) # shape: (nb_steps, nb_outputs)
  norm_factor, _ = torch.max(output_error, dim=0)
  output_error /= norm_factor
  feedback_error = output_error @ feedback_weights # shape: (nb_steps, nb_hidden)

  # Normalize traces:
  norm_factor_e_1, _ = torch.max(eligibility_1, dim=2) # take max along time dimension
  norm_factor_e_2, _ = torch.max(eligibility_2, dim=2)

  norm_factor_e_1 = torch.unsqueeze(norm_factor_e_1, 2) # add an extra dimension for broadcasting for divison below
  norm_factor_e_2 = torch.unsqueeze(norm_factor_e_2, 2)

  eligibility_1 /= norm_factor_e_1 # shape: (nb_inputs, nb_hidden, nb_steps)
  eligibility_2 /= norm_factor_e_2 # shape: (nb_hidden, nb_outputs, nb_steps)

  eligibility_1[eligibility_1 != eligibility_1] = 0 # set NaN values to zero
  eligibility_2[eligibility_2 != eligibility_2] = 0
  
  # Per-parameter learning rate:
  g_ij_sq_1 = torch.sum((feedback_error.T * eligibility_1)**2, dim=2) # shape: (nb_inputs, nb_hidden)
  g_ij_sq_2 = torch.sum((output_error.T * eligibility_2)**2, dim=2) # shape: (nb_hidden, nb_outputs)

  v_ij_1, _ = torch.max(torch.stack([gamma*v_ij_1, g_ij_sq_1], dim=2), dim=2) # shape: (nb_inputs, nb_hidden)
  v_ij_2, _ = torch.max(torch.stack([gamma*v_ij_2, g_ij_sq_2], dim=2), dim=2) # shape: (nb_hidden, nb_outputs)

  learning_rate_1 = initial_rate / torch.sqrt(v_ij_1) # shape: (nb_inputs, nb_hidden)
  learning_rate_2 = initial_rate / torch.sqrt(v_ij_2) # shape: (nb_hidden, nb_outputs)

  # Evaluate Weight Changes:
  w2_change = torch.sum(output_error.T * eligibility_2, dim=2) # sum along time dimension; final shape: (nb_hidden, nb_outputs)
  w1_change = torch.sum(feedback_error.T * eligibility_1, dim=2) # final shape: (nb_inputs, nb_hidden)

  # Update Weights:
  w1 += w1_change * learning_rate_1 
  w2 += w2_change * learning_rate_2


# Store the learned weights
learned_weights = w1, w2
file_name = "weights " + str(datetime.datetime.now())
weight_file = open(filename, 'w')
pickle.dump(learned_weights, weight_file)


