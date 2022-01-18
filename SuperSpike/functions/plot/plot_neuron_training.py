from re import A
from matplotlib import pyplot as plt
import numpy as np
import torch

def plot_loss(loss_rec, location, args, figsize=[15,10], dpi=120):
    """
    Inputs:
        loss_rec
        location - (string) directory path to store file
        args
    Returns:
        stores the loss-over-epochs plot in the given directory
    
    """

    plt.plot(loss_rec)
    plt.title("Loss over epochs")
    plt.xlabel("Epochs (each of 0.5 secs)")
    plt.ylabel("Loss")
    plt.figure(figsize=figsize, dpi=dpi)
    plt.savefig(location + 'loss over epochs.png')
    print("Loss plot saved")
#   plt.show()

def plot_learning_rate_params(learning_rate_params, location, args, figsize=[15, 10], dpi=120):
    """
    Inputs:
        learning_rate_params - tuple of r_ij, v_ij, g_ij2 (in that order, where each is of shape:(nb_input, nb_outputs, nb_epochs)
        location - directory to store plot in (string)
        args
    Returns:
        stores plots for avg & median learning rate and other parameters in the given directory
    """

    nb_inputs = args['nb_inputs']

    r_ij_rec, v_ij_rec, g_ij2_rec = learning_rate_params

    for i in range(nb_inputs): 
        print("Saving plot", i)
        fig, ax = plt.subplots(2, figsize=figsize, dpi=dpi, sharex=True)
        fig.clear()

        # plot avg and median learning rate
        #ax[0].plot(torch.flatten(torch.mean(r_ij_rec, dim=0)[0]), label='Avg. Learning Rate')
        #ax[0].plot(torch.flatten(torch.median(r_ij_rec, dim=0)[0]), label='Median Learning Rate')

        ax[0].plot(r_ij_rec[i], label="learning_rate")
        ax[0].legend(loc='best')

        # plot v_ij, g_ij2
        ax[1].plot(v_ij_rec[i], label="v_ij")
        ax[1].plot(g_ij2_rec[i], label="g_ij2")
        ax[1].legend(loc='best')

        plt.savefig(location + 'neuron ' + str(i) +' learning_rate_params.png')