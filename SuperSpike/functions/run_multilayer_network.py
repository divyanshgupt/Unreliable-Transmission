#@title The Two-layer Network

import torch
import numpy as np
import tqdm as tqdm

import functions

def run_multilayer_network(input_trains, w1, w2, args):
    """
    Runs the spiking neural network with one hidden layer for 
    one epoch (i.e. for the given nb_steps).
    Input:
        input_trains
        w1 - input >> hidden layer weights
        w2 - hidden >> output layer weights
        args:['u_rest', 'thres']
    Returns:
        spk_rec_2 - final output spike train
        spk_rec_1 - spike train from the hidden layer
        mem_rec_2 - membrane potential recording from the output layer
        mem_rec_1 - membrane potential recording from the hidden layer
    """

    nb_inputs = w1.shape[0]
    nb_hidden = w1.shape[1]
    nb_outputs = w2.shape[1]

    u_rest = args['u_rest']
    thres = args['thres']
    dtype = args['dtype']
    device = args['device']
    nb_steps = args['nb_steps']
    beta = args['beta']
    alpha = args['alpha']

    mem_1 = u_rest*torch.ones(nb_hidden, dtype=dtype, device=device)
    syn_1 = torch.zeros(nb_hidden, dtype=dtype, device=device)
    mem_2 = u_rest*torch.ones(nb_outputs, dtype=dtype, device=device)
    syn_2 = torch.zeros(nb_outputs, dtype=dtype, device=device)

    # initialize lists to record values
    mem_rec_1 = []
    mem_rec_2 = []
    spk_rec_1 = []
    spk_rec_2 = []
    
    for t in range(nb_steps):
    # for t in tqdm(range(nb_steps)):
    #   print("\n", "Timestep:", t)
        # Spike
        out_1 = functions.spike_fn(mem_1, thres)
        spk_rec_1.append(out_1)
        inp_2 = out_1 # input to the next layer, shape: (nb_hidden,)

        # Reseting membrane potential upon spike
        reset_1 = torch.zeros(nb_hidden, dtype=dtype, device=device)
        # Note: potential for optimization, could remove for loop
        for i in range(nb_hidden): # loop through individual neurons and set reset values based on past activity
            spk_rec_1_i = [row[i] for row in spk_rec_1] # obtains the spiking activity of neuron of interest
            if t < 50:
                if 1 in spk_rec_1_i:
                    reset_1[i] = 1
            elif 1 in spk_rec_1_i[-50:]:
                reset_1[i] = 1
            else:
                reset_1[i] = 0
        
        # evaluating new membrane potential and synaptic input 
        #print("Weight 1 shape:", w1.shape)
        #print("Input train snapshot shape:", input_trains[:, t].shape)
        weighted_inp_1 = w1.T @ input_trains[:, t] # final shape: (nb_hidden, )
        #print("Weighted input 1 shape:", weighted_inp_1.shape)
        new_mem_1 = (beta*mem_1 + (1 - beta)*syn_1 + (1 - beta)*u_rest)*(1 - reset_1) + (reset_1 * u_rest)
        new_syn_1 = alpha*syn_1 + weighted_inp_1

        mem_rec_1.append(mem_1)

        mem_1 = new_mem_1
        syn_1 = new_syn_1

        # Readout Layer
        reset_2 = torch.zeros(nb_outputs, dtype=dtype, device=device)

        for j in range(nb_outputs):
            spk_rec_2_j = [row[j] for row in spk_rec_2]
        #    print(spk_rec_2_j)
            if t < 50:
                if 1 in spk_rec_2_j:
                    reset_2[j] = 1
            elif 1 in spk_rec_2_j[-50:]:
                reset_2[j] = 1
            else:
                reset_2[j] = 0

        # evaluating new membrane potential and synaptic input
        out_2 = functions.spike_fn(mem_2, thres)
        spk_rec_2.append(out_2)
    #  print("Network Output:", out_2)
    #  print("Spk_Rec 2:", spk_rec_2)

        weighted_inp_2 = w2.T @ inp_2 # final shape: (nb_outputs,)
        new_mem_2 = (beta*mem_2 + (1 - beta)*syn_2 + (1 - beta)*u_rest)*(1 - reset_2) + (reset_2 * u_rest)
        new_syn_2 = alpha*syn_2 + weighted_inp_2 # change input term

        mem_rec_2.append(mem_2)

        mem_2 = new_mem_2
        syn_2 = new_syn_2


    spk_rec_1 = torch.stack(spk_rec_1, dim=1) #final_shape:(nb_hidden, nb_steps)
    spk_rec_2 = torch.stack(spk_rec_2, dim=1) #final_shape:(nb_outputs, nb_steps)
    #spk_rec_2 = torch.flatten(spk_rec_2)
    mem_rec_1 = torch.stack(mem_rec_1, dim=1) #final_shape:(nb_hidden, nb_steps)
    mem_rec_2 = torch.stack(mem_rec_2, dim=1) #final_shape:(nb_outputs, nb_steps)

    # Presynaptic Traces
    ## At hidden layer: 
    presynaptic_traces_1 = functions.presynaptic_trace(input_trains, args) #final_shape: (nb_inputs, nb_steps)
    ## At output layer: 
    presynaptic_traces_2 = functions.presynaptic_trace(spk_rec_1, args) #final_shape: (nb_hidden, nb_steps)

    # Hebbian Coincidence Term:
    ## At hidden layer:
    h_1 = mem_rec_1 - thres #final_shape: (nb_hidden, nb_steps)
    post_1 = 1 / (1 + torch.abs(h_1))**2
    A1 = torch.unsqueeze(presynaptic_traces_1, 1) #final_shape: (nb_inputs, nb_hidden, nb_steps)
    B1 = torch.unsqueeze(post_1, 0) #final_shape: nb_inputs, nb_hidden, nb_steps)
    hebbian_1 = A1 * B1  #final_shape: (nb_inputs, nb_hidden, nb_steps)

    ## At output layer: 
    h_2 = mem_rec_2 - thres
    post_2 = 1 / (1 + torch.abs(h_2))**2
    A2 = torch.unsqueeze(presynaptic_traces_2, 1)
    B2 = torch.unsqueeze(post_2, 0)
    hebbian_2 = A2 * B2 #final_shape: (nb_hidden, nb_outputs, nb_steps)

    # Eligibility Trace (double exponential kernel over the hebbian coincidence term)
    ## At hidden layer:
    eligibility_1 = functions.eligibility_trace(hebbian_1, args) #final_shape: (nb_inputs, nb_hidden, nb)
    ## At output layer:
    eligibility_2 = functions.eligibility_trace(hebbian_2, args) #final_shape: (nb_hidden, nb_outputs, nb_steps)

    return eligibility_1, eligibility_2, presynaptic_traces_1, presynaptic_traces_2, spk_rec_2, spk_rec_1, mem_rec_2, mem_rec_1
