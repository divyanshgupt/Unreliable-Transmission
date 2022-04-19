def plot_weight_distribution(weights):
    """
    Inputs:
        weights: tensor, shape: (nb_inputs, nb_outputs)
    
    """
    weights = torch.flatten(weights).to('cpu')
    plt.hist(weights, bins='fd')
    plt.title("Weights Distribution")
    plt.show()