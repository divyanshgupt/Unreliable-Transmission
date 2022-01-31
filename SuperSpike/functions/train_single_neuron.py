import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import functions
import pickle
import datetime


def train_single_neuron(input_trains, target, weights, r_0, args):
    """

    Inputs:
        nb_epochs - 
        input_trains - input spike trains, shape: (nb_inputs, nb_steps)
        target - target spike train, shape:(nb_steps,)
        weights - shape:(nb_inputs, nb_outputs)
        r_0 - basal learning rate (scalar)
        args
  
    returns:
        trained weights
        loss_rec
        g_ij2_rec
        learning_rate_rec: shape: 
    """

    nb_epochs = args['nb_epochs']
    nb_steps = args['nb_steps']
    nb_inputs = args['nb_inputs']
    nb_outputs = args['nb_outputs']
    device = args['device']
    dtype = args['dtype']
    dt = args['timestep_size']


    loss_rec = np.zeros(nb_epochs)

    v_ij = 1e-8*torch.ones((nb_inputs, nb_outputs), device=device, dtype=dtype)
    print("Initial v_ij coeff = 1e-8")
   # v_ij = torch.zeros((nb_inputs, nb_outputs), device=device, dtype=dtype)
    gamma = float(np.exp(-dt/args['tau_rms']))


    g_ij2_rec = torch.zeros((nb_inputs, nb_outputs, nb_epochs), device=device, dtype=dtype)
    v_ij_rec = torch.zeros((nb_inputs, nb_outputs, nb_epochs), device=device, dtype=dtype)
    r_ij_rec = torch.zeros((nb_inputs, nb_outputs, nb_epochs), device=device, dtype=dtype)


    for i in tqdm(range(nb_epochs)):
        
        mem_rec, spk_rec, error_rec, eligibility_rec, pre_trace_rec = functions.run_single_neuron(input_trains, weights,
                                                                                        target, args)
        loss = functions.new_van_rossum_loss(spk_rec, target, args)
        loss_rec[i] = loss
        print("Loss =", loss)
        # Weight update
        weight_updates = torch.sum(error_rec * eligibility_rec, dim=2)
        assert weight_updates.shape == (nb_inputs, nb_outputs), "wegiht_updates shape incorrect"

        # per-parameter learning rate
    #   g_ij2 = (error_rec * eligibility_rec)[:, :, -1]**2 # this has a problem 
        g_ij2 = torch.sum((error_rec*eligibility_rec)**2, dim=2)
     #   print("NaN values:", g_ij2[g_ij2 != g_ij2])
     #   print("Zeros:", g_ij2[g_ij2==0])
      #  g_ij2[g_ij2 == 0] = 1
        
    #    g_ij2[g_ij2 <= 1e-5] 
        assert g_ij2.shape == (nb_inputs, nb_outputs), "g_ij2 shape incorrect"
        # Question 1: Whether to take the value of g_ij at the last timestep in each epoch or to take the sum of its values over all timesteps in the epoch?
        # Question 2: How to do normalized convolution for error_signal and eligibility_trace?

        #v_ij, _ = torch.max(torch.stack([gamma*v_ij, g_ij2], dim=2), dim=2) # shape: (nb_inputs, nb_outputs)
        v_ij = torch.max(gamma*v_ij, g_ij2)
        assert v_ij.shape == (nb_inputs, nb_outputs), "v_ij shape incorrect"
        
        # Evaluate learning rate for this epoch
        r_ij = r_0 / torch.sqrt(v_ij)   # shape: (nb_inputs, nb_outputs)
        # Store learning rate information for this epoch for xth weight
        g_ij2_rec[:, :, i] = g_ij2 
        v_ij_rec[:, :, i] = v_ij
        r_ij_rec[:, :, i] = r_ij

        rate_med = torch.median(r_ij)
        print("Median Learning Rate:", rate_med)
        rate_mean = torch.mean(r_ij)
        print("Avg. Learning Rate:", rate_mean)

        weights += r_ij * weight_updates

    
    learning_rate_params = (r_ij_rec, v_ij_rec, g_ij2_rec)

    return weights, loss_rec, learning_rate_params
        
    """
    for i in tqdm(range(nb_epochs)):

        #print("\n Iteration:", i+1)
        mem_rec, spk_rec, error_rec, eligibility_rec, pre_trace_rec = functions.run_single_neuron(input_trains, weights,
                                                                                        target, args)
        # loss = van_rossum_loss(spk_rec, target, args)
        # print("Loss = ", loss)

        norm_loss = functions.van_rossum_loss(spk_rec, target, args)
        #print("Normalized Loss =", norm_loss)
        loss_rec[i] = norm_loss

        norm_factor, _ = torch.max(torch.abs(eligibility_rec), dim=2) # take max along time dimension for each i-jth synapse
        norm_factor = torch.unsqueeze(norm_factor, 2)

        eligibility_rec = eligibility_rec / norm_factor # shape: (nb_inputs, nb_outputs, nb_steps)
        eligibility_rec[eligibility_rec != eligibility_rec] = 0
        assert eligibility_rec.shape == (nb_inputs, nb_outputs, nb_steps), "eligibility_rec shape incorrect"

        # Normalizing the error signal:
        norm_factor = torch.max(torch.abs(error_rec))
        error_rec = error_rec / norm_factor  # shape: shape: (nb_steps,)
        assert error_rec.shape == (nb_steps,), "error_rec shape incorrect"

        # Weight update
        weight_updates = torch.sum(error_rec * eligibility_rec, dim=2)
        assert weight_updates.shape == (nb_inputs, nb_outputs), "wegiht_updates shape incorrect"

        # per-parameter learning rate
    #   g_ij2 = (error_rec * eligibility_rec)[:, :, -1]**2 # this has a problem 
        g_ij2 = torch.sum((error_rec*eligibility_rec)**2, dim=2)
        assert g_ij2.shape == (nb_inputs, nb_outputs), "g_ij2 shape incorrect"
        # Question 1: Whether to take the value of g_ij at the last timestep in each epoch or to take the sum of its values over all timesteps in the epoch?
        # Question 2: How to do normalized convolution for error_signal and eligibility_trace?

        v_ij, _ = torch.max(torch.stack([gamma*v_ij, g_ij2], dim=2), dim=2) # shape: (nb_inputs, nb_outputs)
        assert v_ij.shape == (nb_inputs, nb_outputs), "v_ij shape incorrect"
        
        # Evaluate learning rate for this epoch
        r_ij = r_0 / torch.sqrt(v_ij)   # shape: (nb_inputs, nb_outputs)
        # Store learning rate information for this epoch for xth weight
        g_ij2_rec[:, :, i] = g_ij2 
        v_ij_rec[:, :, i] = v_ij
        r_ij_rec[:, :, i] = r_ij

     #   rate_med = torch.median(r_ij)
     #   print("Median Learning Rate:", rate_med)
     #   rate_mean = torch.mean(r_ij)
     #   print("Avg. Learning Rate:", rate_mean)

        weights += r_ij * weight_updates

    """
