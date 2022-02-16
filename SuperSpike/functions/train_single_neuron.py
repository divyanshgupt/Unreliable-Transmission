from re import I
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import functions
import pickle
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

    v_ij = torch.ones((nb_inputs, nb_outputs), device=device, dtype=dtype)
    epsilon = args['epsilon']
    print("Epsilon =", epsilon)
    print("Initial v_ij coeff = 1")
   # v_ij = torch.zeros((nb_inputs, nb_outputs), device=device, dtype=dtype)
    gamma = float(np.exp(-dt/args['tau_rms']))


    g_ij2_rec = torch.zeros((nb_inputs, nb_outputs, nb_epochs), device=device, dtype=dtype)
    v_ij_rec = torch.zeros((nb_inputs, nb_outputs, nb_epochs), device=device, dtype=dtype)
    r_ij_rec = torch.zeros((nb_inputs, nb_outputs, nb_epochs), device=device, dtype=dtype)

    spk_rec = torch.empty((nb_epochs, nb_steps), device=device, dtype=dtype)
    weight_update_rec = torch.empty((nb_epochs, nb_inputs, nb_outputs), device=device, dtype=dtype)
    weight_change_rec = torch.empty((nb_epochs, nb_inputs, nb_outputs), device=device, dtype=dtype)
    weight_rec = torch.empty((nb_epochs, nb_inputs, nb_outputs), device=device, dtype=dtype)

    for i in tqdm(range(nb_epochs)):
        print("\n Iteration no: ", i)
        mem_rec, spk_rec[i], error_rec, eligibility_rec, pre_trace_rec = functions.run_single_neuron(input_trains, weights,
                                                                                        target, args)
        loss = functions.new_van_rossum_loss(spk_rec[i], target, args)
        loss_rec[i] = loss.to('cpu')
        print("Loss =", loss)

        if i >= 1:
            if loss_rec[i] >= 8*loss_rec[i-1]:

                print("\n \n Loss overshot in last update")
                
            #    plt.plot(weight_change, '.')
            #    plt.show()

                print("Previous weight_change vector =", weight_change)
                print("\n Previous r_ij vector =", r_ij)
                print("\n Previous g_ij vector =", g_ij2)
                print("\n Previous v_ij vector =", v_ij)
                print("\n \n")

                # fig, ax = plt.subplots(4, 1, sharex=True)

                # im = ax[0].imshow(torch.unsqueeze(weight_change, 0))
                # ax[0].set_title("Weight change")

                # im = ax[1].imshow(torch.unsqueeze(r_ij, 0))
                # ax[1].set_title("r_ij")

                # im = ax[2].imshow(torch.unsqueeze(g_ij2, 0))
                # ax[2].set_title("g_ij2")

                # im = ax[3].imshow(torch.unsqueeze(v_ij, 0))
                # ax[3].set_title("v_ij")

                # fig.colorbar(im, )
                # fig.suptitle("Underlying parameters")

<<<<<<< HEAD
                """
                image = torch.vstack((weight_change.flatten(), r_ij.flatten(), g_ij2.flatten(), v_ij.flatten()))
                image = torch.vstack((weight_change.flatten(), r_ij.flatten()))
                # print("Image array = ", image)
                # print(image.shape)
=======

            #    image = torch.vstack((weight_change.flatten(), r_ij.flatten(), g_ij2.flatten(), v_ij.flatten()))
            #    image = torch.vstack((weight_change.flatten(), r_ij.flatten()))
                # image = torch.cat((weight_change.flatten(), r_ij.flatten()), dim=1)

                # # print("Image array = ", image)
                # # print(image.shape)
>>>>>>> 5a151024b187fb22f38c5a1ef1d8ec888c5340fb
                
                # ax = plt.subplot(211)
                # im = ax.imshow(image, aspect='auto')
                # # Align colorbar:            
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes('right', size='5%', pad=0.05)

                # ax.set_yticks([0, 1, 2, 3])
                # ax.set_yticklabels(['weight_change', 'r_ij', 'g_ij2', 'v_ij'])
                
                # plt.colorbar(im, cax=cax)

                # plt.show()
                
<<<<<<< HEAD
                plt.figure()
                fig_1, axs = plt.subplots(2, 1, sharex=True, figsize = (15,10), dpi=120)

                positions = np.arange(0, nb_steps)
                spike_positions_prev = positions[spk_rec[i-1] == 1]
                axs[0].eventplot(spike_positions_prev)
                axs[0].set_title("Previous Spike Train")
                axs[0].set_xlim([0, nb_steps])

                positions = np.arange(0, nb_steps)
                spike_positions_curr = positions[spk_rec[i] == 1]
                axs[1].eventplot(spike_positions_curr)
                axs[1].set_title("Current Spike Train")
                axs[1].set_xlim([0, nb_steps])
                fig_1.show()
                """
=======
                # plt.figure()
                # fig_1, axs = plt.subplots(2, 1, sharex=True, figsize = (15,10), dpi=120)

                # positions = np.arange(0, nb_steps)
                # spike_positions_prev = positions[spk_rec[i-1] == 1]
                # axs[0].eventplot(spike_positions_prev)
                # axs[0].set_title("Previous Spike Train")
                # axs[0].set_xlim([0, nb_steps])

                # positions = np.arange(0, nb_steps)
                # spike_positions_curr = positions[spk_rec[i] == 1]
                # axs[1].eventplot(spike_positions_curr)
                # axs[1].set_title("Current Spike Train")
                # axs[1].set_xlim([0, nb_steps])
                # fig_1.show()

>>>>>>> 5a151024b187fb22f38c5a1ef1d8ec888c5340fb



        # Weight update
        weight_updates = torch.sum(error_rec * eligibility_rec, dim=2)
        assert weight_updates.shape == (nb_inputs, nb_outputs), "wegiht_updates shape incorrect"



        # per-parameter learning rate
    #   g_ij2 = (error_rec * eligibility_rec)[:, :, -1]**2 # this has a problem 
        g_ij2 = torch.sum((error_rec*eligibility_rec)**2, dim=2) # this summing after squaring is likely wrong
    #    g_ij2 = torch.sum((error_rec*eligibility_rec), dim=2)**2
     #   print("NaN values:", g_ij2[g_ij2 != g_ij2])
     #   print("Zeros:", g_ij2[g_ij2==0])
      #  g_ij2[g_ij2 == 0] = 1
        
        
        # print(g_ij2[g_ij2 <= 1e-5]) 
        assert g_ij2.shape == (nb_inputs, nb_outputs), "g_ij2 shape incorrect"
        # Question 1: Whether to take the value of g_ij at the last timestep in each epoch or to take the sum of its values over all timesteps in the epoch?
        # Question 2: How to do normalized convolution for error_signal and eligibility_trace?

        v_ij, _ = torch.max(torch.stack([gamma*v_ij, g_ij2], dim=2), dim=2) # shape: (nb_inputs, nb_outputs)
        #v_ij = torch.max(gamma*v_ij, g_ij2)
        assert v_ij.shape == (nb_inputs, nb_outputs), "v_ij shape incorrect"
        
        # Evaluate learning rate for this epoch
        r_ij = r_0 / torch.sqrt(v_ij + epsilon)   # shape: (nb_inputs, nb_outputs)
        # Store learning rate information for this epoch for xth weight
        g_ij2_rec[:, :, i] = g_ij2 
        v_ij_rec[:, :, i] = v_ij
        r_ij_rec[:, :, i] = r_ij

        rate_med = torch.median(r_ij)
        print("Median Learning Rate:", rate_med)
        rate_mean = torch.mean(r_ij)
        print("Avg. Learning Rate:", rate_mean)

        weight_change = r_ij * weight_updates

        weights += weight_change

        weight_update_rec[i] = weight_updates   # record weight-update
        weight_change_rec[i] = weight_change
        weight_rec[i] = weights # record weights at the end of epoch
        # print("Weight norm = ", torch.norm(weights))
        


    
    learning_rate_params = (r_ij_rec, v_ij_rec, g_ij2_rec)

    # Store the recordings
    recordings = (spk_rec, weight_update_rec, weight_change_rec, weight_rec, r_ij_rec, v_ij_rec, g_ij2_rec)


    return weights, loss_rec, recordings
        
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
