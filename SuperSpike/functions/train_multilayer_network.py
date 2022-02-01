import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import functions

def train_multilayer_network(input_trains, w1, w2, feedback_weights, target, r_0, args):
    """
    Trains a three-layer network
    Inputs:
        input_trains:
        w1
        w2
        target
        args
    Returns:    
    """
    
    device = args['device']
    dtype = args['dtype']
    nb_inputs = args['nb_inputs']
    nb_hidden = args['nb_hidden']
    nb_outputs = args['nb_outputs']
    dt = args['timestep_size']
    nb_steps = args['nb_steps']
    nb_epochs = args['nb_epochs']

    gamma = float(np.exp(-dt/args['tau_rms']))

    loss_rec = np.zeros(nb_epochs)

    v_ij_1 = 1e-10*torch.ones((nb_inputs, nb_hidden), device=device, dtype=dtype)
    v_ij_2 = 1e-10*torch.ones((nb_hidden, nb_outputs), device=device, dtype=dtype)

    for i in tqdm(range(nb_epochs)):
        print("Epoch no:", i)
        eligibility_1, eligibility_2, presynaptic_trace_1, presynaptic_traces_2, spk_rec_2, spk_rec_1, mem_rec_2, mem_rec_1 = functions.run_multilayer_network(input_trains, w1, w2, args)
        
        output = torch.flatten(spk_rec_2)

        # Evaluate Loss & Store for Plotting:
        loss = functions.new_van_rossum_loss(target, output, args)

        loss_rec[i] = loss
        print("Loss =", loss)

        # Evaluate Error Signal for both layers:
        output_error = torch.unsqueeze(functions.new_error_signal(output, target, args), 1) # shape: (nb_steps, nb_outputs)
    #    norm_factor, _ = torch.max(output_error, dim=0)
    #    output_error /= norm_factor
        feedback_error = output_error @ feedback_weights # shape: (nb_steps, nb_hidden)

        # Normalize traces:
    #    norm_factor_e_1, _ = torch.max(eligibility_1, dim=2) # take max along time dimension
    #    norm_factor_e_2, _ = torch.max(eligibility_2, dim=2)

    #    norm_factor_e_1 = torch.unsqueeze(norm_factor_e_1, 2) # add an extra dimension for broadcasting for divison below
    #    norm_factor_e_2 = torch.unsqueeze(norm_factor_e_2, 2)

    #    eligibility_1 /= norm_factor_e_1 # shape: (nb_inputs, nb_hidden, nb_steps)
    #    eligibility_2 /= norm_factor_e_2 # shape: (nb_hidden, nb_outputs, nb_steps)

    #    eligibility_1[eligibility_1 != eligibility_1] = 0 # set NaN values to zero
    #    eligibility_2[eligibility_2 != eligibility_2] = 0
        
        # Per-parameter learning rate:
        g_ij_sq_1 = torch.sum((feedback_error.T * eligibility_1)**2, dim=2) # shape: (nb_inputs, nb_hidden)
        g_ij_sq_2 = torch.sum((output_error.T * eligibility_2)**2, dim=2) # shape: (nb_hidden, nb_outputs)

        v_ij_1, _ = torch.max(torch.stack([gamma*v_ij_1, g_ij_sq_1], dim=2), dim=2) # shape: (nb_inputs, nb_hidden)
        v_ij_2, _ = torch.max(torch.stack([gamma*v_ij_2, g_ij_sq_2], dim=2), dim=2) # shape: (nb_hidden, nb_outputs)

        learning_rate_1 = r_0 / torch.sqrt(v_ij_1) # shape: (nb_inputs, nb_hidden)
        learning_rate_2 = r_0 / torch.sqrt(v_ij_2) # shape: (nb_hidden, nb_outputs)

        # Evaluate Weight Changes:
        w2_change = torch.sum(output_error.T * eligibility_2, dim=2) # sum along time dimension; final shape: (nb_hidden, nb_outputs)
        w1_change = torch.sum(feedback_error.T * eligibility_1, dim=2) # final shape: (nb_inputs, nb_hidden)

        # Update Weights:
        w1 += w1_change * learning_rate_1 
        w2 += w2_change * learning_rate_2

    return w1, w2, loss_rec
