def plot_single_train(spike_train, nb_steps, timestep_size, idx=0):

    positions = np.arange(0, nb_steps)
    spike_positions = positions[spike_train == 1]
    # print(spike_positions)
    plt.eventplot(spike_positions, lineoffsets=idx)
    plt.xlim(0, nb_steps)
    #plt.show()

