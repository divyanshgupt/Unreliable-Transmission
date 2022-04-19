def plot_spike_trains(spike_trains, title='Spike Trains'):
    """
    
    """
    plt.figure(dpi = 100)
    for i in range(len(spike_trains)):
        plot_single_train(spike_trains[i], nb_steps, timestep_size, idx=i)
    plt.title(title)
    plt.xlabel('Timestep')
    plt.ylabel('Spike Train No.')
    plt.show()
