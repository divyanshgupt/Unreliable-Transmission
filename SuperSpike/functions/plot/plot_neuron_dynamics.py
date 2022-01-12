def plot_neuron_dynamics(mem_rec, spk_rec, error_rec, target, figsize=(15, 10), dpi=120):

    """
    Plots target train, error signal, output train and membrane potential from top to bottom in that order
    Inputs:
        mem_rec: 
        spk_rec:
        error_rec:
        target: 
    """

    fig, axs = plt.subplots(4, sharex=True, figsize=figsize, dpi=dpi)

    ## Plot the target spike train
    positions = np.arange(0, nb_steps)
    spike_positions = positions[target == 1]
    axs[0].eventplot(spike_positions)
    axs[0].set_title("Target Spike Train")
    axs[0].set_xlim([0, nb_steps])

    ## Plot error signal
    axs[1].plot(error_rec)
    axs[1].set_title("Error Signal")

    ## Plot output spike train
    positions = np.arange(0, nb_steps)
    spike_positions = positions[spk_rec == 1]
    axs[2].eventplot(spike_positions)
    axs[2].set_title("Output Spike Train")
    axs[2].set_xlim([0, nb_steps])

    ## Plot membrane potential
    axs[3].plot(mem_rec)
    axs[3].set_title("Membrane Potential")
    axs[3].set_ylabel("Potential (in mV)")

    for ax in axs.flat:
        ax.set(xlabel = "Timestep")
        # Hide x labels and tick labels for top plots and y ticks for right plots:
        ax.label_outer()