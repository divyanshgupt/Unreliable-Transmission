def initialize_weights(nb_inputs, nb_outputs, scale=70):
    """
    Inputs:
        nb_inputs - 
        nb_outputs - 
        scale - 
    Returns:
        weights: shape:(nb_inputs, nb_outputs)
    """
    weight_scale = scale*(1 - 0) # copied from spytorch

    weights = torch.empty((nb_inputs, nb_outputs), device=device, dtype=dtype)
    weights = torch.nn.init.normal_(weights, mean=0.5, std=weight_scale/np.sqrt(nb_inputs))
    print("Weight initialization done")
    return weights