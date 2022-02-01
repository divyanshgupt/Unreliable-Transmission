import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import datetime
import pickle
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
        'nb_epochs': 100
        } 
    
args['device'] = device
args['dtype'] = dtype
nb_inputs = args['nb_inputs']
nb_hidden = args['nb_hidden']
nb_outputs = args['nb_outputs']

nb_steps = args['nb_steps']
dt = args['timestep_size']

tau_syn = args['tau_syn']
tau_mem = args['tau_mem']

alpha = np.exp(-dt/tau_syn)
beta = np.exp(-dt/tau_mem)

args['alpha'] = alpha
args['beta'] = beta

# 100 independent poisson trains
input_trains = functions.poisson_trains(nb_inputs, 10*np.ones(100), args)

# 5 equidistant spikes over 500 msec.
target = torch.zeros((nb_steps), device=device, dtype=dtype)
target[500::int(nb_steps/5)] = 1

#@title Training the network

#weight_scale = 100

#w1, w2 = functions.initialize_weights_multilayer(nb_inputs, nb_hidden, nb_outputs, args, weight_scale)

w1, w2 = functions.new_initialize_weights_multilayer(nb_inputs, nb_hidden, nb_outputs, args)


feedback_weights = functions.random_feedback(nb_hidden, nb_outputs, args).T # shape: (nb_outputs, nb_hidden)
loss_rec = []

v_ij_1 = 1e-10*torch.ones((nb_inputs, nb_hidden), device=device, dtype=dtype)
v_ij_2 = 1e-10*torch.ones((nb_hidden, nb_outputs), device=device, dtype=dtype)


gamma = float(np.exp(-dt/args['tau_rms']))

learning_rates = np.array([5, 1, 10, 0.5, 0.1]) * 1e-3


for r_0 in learning_rates:
  print("Learning rate =", r_0)
  new_w1, new_w2, loss_rec = functions.train_multilayer_network(input_trains, w1, w2, feedback_weights, target, r_0, args)
  
  plt.plot(loss_rec)
  plt.title("Loss over epochs, learning rate = " + str(r_0))
  plt.show()







# Store the learned weights
learned_weights = w1, w2
file_name = "weights " + str(datetime.datetime.now())
weight_file = open(file_name, 'w')
pickle.dump(learned_weights, weight_file)


