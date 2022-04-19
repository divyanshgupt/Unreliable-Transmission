import numpy as np

args = {'thres': -50,
        'u_rest': -60,
        'tau_mem': 1e-2,
        'tau_syn': 5e-3,
        'tau_ref': 5e-3,
        'tau_het': 5e-3, # regulation term time constant (pure guess)
        't_rise': 5e-3, # the pre-synaptic double exponential kernel rise time
        't_decay': 1e-2, # the pre-synaptic double exponential kernel decay time
        'timestep_size': 1e-4, # 0.1 msec
        't_rise_alpha': 5e-3, # change this
        't_decay_alpha': 1e-2, # change this 
        'nb_steps': 5000, # 0.5 secs in total
        'tau_rms': 5e-4,
        'nb_inputs': 100,
        'nb_hidden': 4,
        'nb_outputs': 1,
        'nb_epochs': 1000,
        'rho': 1, # regularization strength
        'a': 4, # exponent parameter for regularization evaluation
        'epsilon': 1e-4
        } 
    
args['device'] = device
args['dtype'] = dtype
nb_inputs = args['nb_inputs']
nb_hidden = args['nb_hidden']
nb_outputs = args['nb_outputs']

nb_steps = args['nb_steps']
nb_epochs = args['nb_epochs']
dt = args['timestep_size']

tau_syn = args['tau_syn']
tau_mem = args['tau_mem']

alpha = np.exp(-dt/tau_syn)
beta = np.exp(-dt/tau_mem)

args['alpha'] = alpha
args['beta'] = beta
