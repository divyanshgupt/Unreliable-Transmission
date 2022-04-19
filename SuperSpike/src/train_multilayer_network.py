import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
#from SuperSpike.functions.error_signal import error_signal
import src

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
    rho = args['rho'] # regularization strength
    epsilon = args['epsilon']

    gamma = float(np.exp(-dt/args['tau_rms']))

    loss_rec = np.zeros(nb_epochs)

    v_ij_1 = 1e-10*torch.ones((nb_inputs, nb_hidden),    device=device, dtype=dtype)   
    v_ij_2 = 1e-10*torch.ones((nb_hidden, nb_outputs), device=device, dtype=dtype)
    v_ij_1_rec = torch.empty((nb_epochs, nb_inputs, nb_hidden), device=device, dtype=dtype)
    v_ij_2_rec = torch.empty((nb_epochs, nb_hidden, nb_outputs), device=device, dtype=dtype)
    g_ij2_1_rec = torch.empty((nb_epochs, nb_inputs, nb_hidden), device=device, dtype=dtype)
    g_ij2_2_rec = torch.empty((nb_epochs, nb_hidden, nb_outputs), device=device, dtype=dtype)
    learning_rate_1_rec = torch.empty((nb_epochs, nb_inputs, nb_hidden), device=device, dtype=dtype)
    learning_rate_2_rec = torch.empty((nb_epochs, nb_hidden, nb_outputs), device=device, dtype=dtype)

    weight_change_1_rec = torch.empty((nb_epochs, nb_inputs, nb_hidden), device=device, dtype=dtype)
    weight_change_2_rec = torch.empty((nb_epochs, nb_hidden, nb_outputs), device=device, dtype=dtype)
    weight_update_1_rec = torch.empty((nb_epochs, nb_inputs, nb_hidden), device=device, dtype=dtype)
    weight_update_2_rec = torch.empty((nb_epochs, nb_hidden, nb_outputs), device=device, dtype=dtype)

    spk_rec_1 = torch.empty((nb_epochs, nb_hidden, nb_steps), device=device, dtype=dtype)
    spk_rec_2 = torch.empty((nb_epochs, nb_outputs, nb_steps), device=device, dtype=dtype)
    mem_rec_1 = torch.empty((nb_epochs, nb_hidden, nb_steps), device=device, dtype=dtype)
    mem_rec_2 = torch.empty((nb_epochs, nb_outputs, nb_steps), device=device, dtype=dtype)
    eligibility_1 = torch.empty((nb_epochs, nb_inputs, nb_hidden, nb_steps), device=device, dtype=dtype)
    eligibility_2 = torch.empty((nb_epochs, nb_hidden, nb_outputs, nb_steps), device=device, dtype=dtype)

    presynaptic_traces_1 = torch.empty((nb_epochs, nb_inputs, nb_steps), device=device, dtype=dtype)
    presynaptic_traces_2 = torch.empty((nb_epochs, nb_hidden, nb_steps), device=device, dtype=dtype)

    output_error = torch.empty((nb_epochs, nb_steps, nb_outputs), device=device, dtype=dtype)
    feedback_error = torch.empty((nb_epochs, nb_steps, nb_hidden), device=device, dtype=dtype)

    w1_rec = torch.empty((nb_epochs + 1, nb_inputs, nb_hidden), device=device, dtype=dtype)
    w2_rec = torch.empty((nb_epochs + 1, nb_hidden, nb_outputs), device=device, dtype=dtype)

    w1_rec[0] = w1
    w2_rec[0] = w2
    for i in tqdm(range(nb_epochs)):
        
        print("Epoch no:", i)
        eligibility_1[i], eligibility_2[i], presynaptic_traces_1[i], presynaptic_traces_2[i], spk_rec_2[i], spk_rec_1[i], mem_rec_2[i], mem_rec_1[i] = src.run_multilayer_network(input_trains, w1, w2, args)
        
        output = torch.flatten(spk_rec_2[i])

        # Evaluate Loss & Store for Plotting:
        loss = src.new_van_rossum_loss(target, output, args)

        loss_rec[i] = loss
        print("Loss =", loss)

        # Evaluate Error Signal for both layers:
        output_error[i] = torch.unsqueeze(src.new_error_signal(output, target, args), 1) # shape: (nb_steps, nb_outputs)
    #    norm_factor, _ = torch.max(output_error, dim=0)
    #    output_error /= norm_factor
        feedback_error[i] = output_error[i] @ feedback_weights # shape: (nb_steps, nb_hidden)

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
        g_ij_sq_1 = torch.sum((feedback_error[i].T * eligibility_1[i])**2, dim=2) # shape: (nb_inputs, nb_hidden)
        g_ij_sq_2 = torch.sum((output_error[i].T * eligibility_2[i])**2, dim=2) # shape: (nb_hidden, nb_outputs)

        v_ij_1, _ = torch.max(torch.stack([gamma*v_ij_1, g_ij_sq_1], dim=2), dim=2) # shape: (nb_inputs, nb_hidden)
        v_ij_2, _ = torch.max(torch.stack([gamma*v_ij_2, g_ij_sq_2], dim=2), dim=2) # shape: (nb_hidden, nb_outputs)

        learning_rate_1 = r_0 / torch.sqrt(v_ij_1 + epsilon) # shape: (nb_inputs, nb_hidden)
        learning_rate_2 = r_0 / torch.sqrt(v_ij_2 + epsilon) # shape: (nb_hidden, nb_outputs)

        learning_rate_1_rec[i] = learning_rate_1
        learning_rate_2_rec[i] = learning_rate_2
        v_ij_1_rec[i] = v_ij_1
        v_ij_2_rec[i] = v_ij_2
        g_ij2_1_rec[i] = g_ij_sq_1
        g_ij2_2_rec[i] = g_ij_sq_2


        # Evaluate Weight change:
        w2_change = torch.sum(output_error[i].T * eligibility_2[i], dim=2) # sum along time dimension; final shape: (nb_hidden, nb_outputs)
        w1_change = torch.sum(feedback_error[i].T * eligibility_1[i], dim=2) # final shape: (nb_inputs, nb_hidden)

        weight_change_1_rec[i] = w1_change
        weight_change_2_rec[i] = w2_change

        regularization_1 = src.heterosynaptic_regularization(spk_rec_1[i], args) # shape: (nb_hidden,)
        regularization_2 = src.heterosynaptic_regularization(spk_rec_2[i], args) # shape: (nb_outputs,)

        # Update Weights:
        weight_update_1 = (w1_change * learning_rate_1) - rho*(w1*regularization_1)
        weight_update_2 = (w2_change * learning_rate_2) - rho*(w2*regularization_2)

        w1 += weight_update_1
        w2 += weight_update_2

        weight_update_1_rec[i] = weight_update_1
        weight_update_2_rec[i] = weight_update_2

        rate_med_1 = torch.median(learning_rate_1)
        rate_mean_1 = torch.mean(learning_rate_1)
        print("Learning rate 1: Median =", rate_med_1)
        print("Learning rate 1: Mean =", rate_mean_1)
        
        rate_med_2 = torch.median(learning_rate_2)
        rate_mean_2 = torch.mean(learning_rate_2)
        print("Learning rate 2: Median =", rate_med_2)
        print("Learning rate 2: Mean =", rate_mean_2)

        w1_rec[i + 1] = w1
        w2_rec[i + 1] = w2 

    neural_dynamics = (spk_rec_1, spk_rec_2, mem_rec_1, mem_rec_2, presynaptic_traces_1, presynaptic_traces_2, eligibility_1, eligibility_2, output_error, feedback_error)  
    weight_dynamics = (w1_rec, w2_rec, feedback_weights, weight_change_1_rec, weight_change_2_rec, weight_update_1_rec, weight_update_2_rec)
    learning_rate_dynamics = (learning_rate_1_rec, learning_rate_2_rec, v_ij_1_rec, v_ij_2_rec, g_ij2_1_rec, g_ij2_2_rec)
    recordings = (args, input_trains, neural_dynamics, weight_dynamics, learning_rate_dynamics)


    return w1, w2, loss_rec, recordings
