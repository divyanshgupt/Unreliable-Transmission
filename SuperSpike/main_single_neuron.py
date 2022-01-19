import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import os # To create a folder for each run
import functions 
#import functions.plot
import datetime
import json # To save simulation parameters in a text file
import pdb
import pickle

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
        'nb_epochs': 4
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

# Create Target Train
target = torch.zeros(nb_steps, device=device, dtype=dtype)
target[:: nb_steps//5] = 1

weights = functions.initialize_weights(nb_inputs, nb_outputs, args, scale=80) # initialize weights

# v_ij = 1e-2*torch.zeros((nb_inputs, nb_outputs), device=device, dtype=dtype)
learning_rates = np.array([10, 5, 1, 0.5, 0.1]) * 1e-3

for r_0 in learning_rates:
    #r_0 = 5e-3 # basal learning rate

    new_weights, loss_rec, learning_rate_params = functions.train_single_neuron(input_trains, target, weights, r_0, args)
    #r_ij, v_ij, g_ij2 = learning_rate_params
    
    data_folder = "data/" + str(datetime.datetime.today())[:10] + ' rate = ' + str(r_0) + '/'
    #os.makedirs(location)
    location = os.path.abspath(data_folder)
    location = os.path.join(os.getcwd(), location)
    print(location)
#    data_folder = "/data/rate = " + str(r_0) + ' /'
#    location = os.path.join(os.path.abspath(), data_folder)
#    print(location)
#    break
    os.makedirs(location)
    #location = ''
    #location = os.path.cudir()
    # Store parameters (in dict args) in the given location
    functions.plot.plot_loss(loss_rec, location, args) # Saves the loss-over-epochs plot in the given location
    functions.plot.plot_learning_rate_params(learning_rate_params, location, args)

    # Store weights
    file_name = location + "weights" #+ str(datetime.datetime.now())[:10]
    weight_file = open(file_name, 'w')
    pickle.dump(new_weights, weight_file)

    # Store loss_rec
    file_name = location + "loss_rec" #+ str(datetime.datetime.now())
    weight_file = open(file_name, 'w')
    pickle.dump(loss_rec, weight_file)

    # Store learning_rate_params:
    file_name = location + "learning_rate_params" #+ str(datetime.datetime.now())
    param_file = open(file_name, 'w')
    pickle.dump(learning_rate_params, param_file)

    # Store args:
    file_name = location + "args" #+ str(datetime.datetime.now())
    args_file = open(file_name, 'w')
    json.dump(args, args_file)


"""plt.plot(loss_rec)
plt.title("Loss over epochs")
plt.show()

plt.plot(torch.flatten(torch.mean(r_ij, dim=0)[0]), label='Avg. Learning Rate')
plt.plot(torch.flatten(torch.median(r_ij, dim=0)[0]), label='Median Learning Rate')
plt.title("Learning Rate over Epochs")
plt.show()

"""

""" for i in range(nb_epochs):

    print("\n Iteration:", i+1)
    mem_rec, spk_rec, error_rec, eligibility_rec, pre_trace_rec = functions.run_single_neuron(input_trains, weights,
                                                                                    target, args)
    # loss = van_rossum_loss(spk_rec, target, args)
    # print("Loss = ", loss)
    norm_loss = functions.van_rossum_loss(spk_rec, target, args)
    print("Normalized Loss =", norm_loss)
    loss_rec.append(norm_loss)

    norm_factor, _ = torch.max(torch.abs(eligibility_rec), dim=2) # take max along time dimension for each i-jth synapse
    norm_factor = torch.unsqueeze(norm_factor, 2)

    eligibility_rec = eligibility_rec / norm_factor # shape: (nb_inputs, nb_outputs, nb_steps)
    eligibility_rec[eligibility_rec != eligibility_rec] = 0
    assert eligibility_rec.shape == (nb_inputs, nb_outputs, nb_steps), "eligibility_rec shape incorrect"

    # normalizing the error signal:
    norm_factor = torch.max(torch.abs(error_rec))
    error_rec = error_rec / norm_factor  # shape: shape: (nb_steps,)
    assert error_rec.shape == (nb_steps,), "error_rec shape incorrect"
    #print(error_rec)

    # Weight update
    weight_updates = torch.sum(error_rec * eligibility_rec, dim=2)
    assert weight_updates.shape == (nb_inputs, nb_outputs), "wegiht_updates shape incorrect"

    # per-parameter learning rate

    gamma = float(np.exp(-dt/args['tau_rms']))
#   g_ij2 = (error_rec * eligibility_rec)[:, :, -1]**2 # this has a problem 
    g_ij2 = torch.sum((error_rec*eligibility_rec)**2, dim=2)
    assert g_ij2.shape == (nb_inputs, nb_outputs), "g_ij2 shape incorrect"
    # Question 1: Whether to take the value of g_ij at the last timestep in each epoch or to take the sum of its values over all timesteps in the epoch?
    # Question 2: How to do normalized convolution for error_signal and eligibility_trace?

    v_ij = torch.max(gamma*v_ij, g_ij2)

    # Store learning rate information for this epoch for xth weight
    g_ij2_rec.append(g_ij2[x]) 
    v_ij_rec.append(v_ij[x])

    # Evaluate learning rate for this epoch
    r_ij = r_0 / torch.sqrt(v_ij)

    r_ij_rec.append(r_ij[x])

    rate_med = torch.median(r_ij)
    print("Median Learning Rate:", rate_med)
    rate_mean = torch.mean(r_ij)
    print("Avg. Learning Rate:", rate_mean)
    
    print("Rate of 20th weight:", r_ij[19])
    print("Rate of 67th weight:", r_ij[66])

    weights += r_ij * weight_updates
 """

""" fig, ax = plt.subplots(2, sharex=True)

ax[0].plot(loss_rec)
ax[0].set_title("Loss over epochs")
ax[0].set_ylabel("Loss")

#print("Plotting learning rate parameters")
ax[1].plot(v_ij_rec, label='v_ij')
ax[1].plot(g_ij2_rec, label='g_ij2')
ax[1].plot(r_ij_rec, label='r_ij')
ax[1].set_title("Learning rate parameters for the " + str(x) + 'th neuron') """

