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
    
    r_ij, v_ij, g_ij2 = learning_rate_params
    
    fig, ax = plt.subplots(2, figsize=figsize, dpi=dpi, sharex=True)

    # plot avg and median learning rate
    ax[0].plot(torch.flatten(torch.mean(r_ij, dim=0)[0]), label='Avg. Learning Rate')
    ax[0].plot(torch.flatten(torch.median(r_ij, dim=0)[0]), label='Median Learning Rate')

    # plot v_ij, g_ij2


    plt.savefig(location + 'learning_rate_params.png')