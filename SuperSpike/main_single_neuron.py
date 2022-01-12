import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

import functions 

# set device
dtype = torch.float
cpu = torch.device("cpu")
gpu = torch.device("cuda:0")

device = cpu
# Uncomment the line below to run on GPU
#device = gpu


# Set parameters
args = {'thres': -50,
        'u_rest': -60,
        'tau_mem': 1e-2,
        'tau_syn': 5e-3,
        'tau_ref': 5e-3,
        't_rise': 5e-3, # the pre-synaptic double exponential kernel rise time
        't_decay': 1e-2, # the pre-synaptic double exponential kernel decay time
        'timestep_size': 1e-4, # 0.1 msec timesteps
        't_rise_alpha': 5e-3,
        't_decay_alpha': 1e-2,
        'nb_steps': 5000,
        'tau_rms': 5e-3, # this is a guess and might need changing
        'nb_inputs': 100,
        'nb_outputs': 1,
        'device': device, # for functions in different modules
        'dtype': dtype,
        'nb_epochs': 10
        } 


# constants
nb_inputs = args['nb_inputs']
nb_outputs = args['nb_outputs']

nb_steps = args['nb_steps']
dt = args['timestep_size']

tau_syn = args['tau_syn']
tau_mem = args['tau_mem']

alpha = np.exp(-dt/tau_syn)
beta = np.exp(-dt/tau_mem)

args['alpha'] = alpha
args['beta'] = beta

#input trains
spk_freq = 10 # not sure about this, but assuming it since the paper uses 10 Hz frequency as the target output frequency (actually, 5 equidistant spikes over 500 ms)
input_trains = functions.poisson_trains(100, spk_freq*np.ones(100), args)

target = torch.zeros(nb_steps, device=device, dtype=dtype)
target[:: nb_steps//5] = 1


weights = functions.initialize_weights(nb_inputs, nb_outputs, args, scale=80)


# Train neuron

nb_epochs = args['nb_epochs']
loss_rec = []
v_ij = 1e-2*torch.ones((nb_inputs, nb_outputs), device=device, dtype=dtype)
r_0 = 5e-3 # basal learning rate

for i in range(nb_epochs):

    print("Iteration:", i+1)
    mem_rec, spk_rec, error_rec, eligibility_rec, pre_trace_rec = functions.run_single_neuron(input_trains, weights,
                                                                                    target, args)
    # loss = van_rossum_loss(spk_rec, target, args)
    # print("Loss = ", loss)
    norm_loss = functions.van_rossum_loss(spk_rec, target, args)
    print("Normalized Loss =", norm_loss)
    loss_rec.append(norm_loss)

    norm_factor, _ = torch.max(torch.abs(eligibility_rec), dim=2) # take max along time dimension for each i-jth synapse
    norm_factor = torch.unsqueeze(norm_factor, 2)

    eligibility_rec = eligibility_rec / norm_factor
    eligibility_rec[eligibility_rec != eligibility_rec] = 0

    # normalizing the error signal:
    norm_factor = torch.max(torch.abs(error_rec))
    error_rec = error_rec / norm_factor

    # Weight update
    weight_updates = torch.sum(error_rec * eligibility_rec, dim=2)

    # per-parameter learning rate
    gamma = float(np.exp(-dt/args['tau_rms']))
    g_ij2 = (error_rec * eligibility_rec)[:, :, -1]**2
    v_ij = torch.max(gamma*v_ij, g_ij2)

    r_ij = r_0 / torch.sqrt(v_ij)

    weights += r_ij * weight_updates