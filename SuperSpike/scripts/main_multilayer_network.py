import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from datetime import datetime
import pickle
import src
import os
# from src.params_multilayer import *
from params_multilayer import *

note = ""

nb_trials = 1

# Input - 100 independent poisson trains
spk_freq = 10
input_trains = src.poisson_trains(nb_inputs, spk_freq*np.ones(nb_inputs), args)

# Target - 5 equidistant spikes over 500 msec.
target = torch.zeros((nb_steps), device=device, dtype=dtype)
target[500::int(nb_steps/5)] = 1

#w1, w2 = src.initialize_weights_multilayer(nb_inputs, nb_hidden, nb_outputs, args, weight_scale)

w1, w2 = src.new_initialize_weights_multilayer(nb_inputs, nb_hidden, nb_outputs, args) #shape: (nb_inputs, nb_hidden), (nb_hidden, nb_outputs)

feedback_type = 'random'

# Random feedback
#feedback_weights = src.random_feedback(nb_hidden, nb_outputs, args).T # shape: (nb_outputs, nb_hidden)

# Uniform Feedback
#feedback_weights = torch.ones((nb_outputs, nb_hidden), device=device, dtype=dtype)

# Symmetric Feedback
feedback_weights = w2.T


#learning_rates = np.array([50, 5, 1, 10, 0.5, 0.1]) * 1e-3
learning_rates = np.array([50])*1e-3

for r_0 in learning_rates:
  print("Learning rate =", r_0)
  
#  loss_rec = torch.empty((nb_trials, nb_epochs), device=device, dtype=dtype)
  loss_rec = np.empty((nb_trials, nb_epochs))
  recordings_list = []

  for i in range(nb_trials):

    new_w1, new_w2, loss_rec[i], recordings = src.train_multilayer_network(input_trains, w1, w2, feedback_weights, target, r_0, args)
    recordings_list.append(recordings)  
    plt.plot(loss_rec[i], alpha=0.5)

  plt.plot(np.mean(loss_rec, axis=0), color='black', label='Avg. Loss')
  plt.title("Loss over epochs, learning rate = " + str(r_0))
  plt.legend()
  # plt.show()

  date_stamp = str(datetime.today())[:13]
  location = src.set_location(f'../data/multilayer/{feedback_type}/{date_stamp} r_0={r_0}')


  # plt.savefig(location + "/loss over epochs" + "learning-rate = " + str(r_0) +
  #             ", epsilon = " + str(args['epsilon']) + "spike freq = " + str(spk_freq) + ".png")

  plt.savefig(f'{location}/loss over epochs - r_0={r_0} - epsilon={args["epsilon"]} - spk_freq={spk_freq}.png')

  src.save_data(f'{args} \n \n {note}', location, f'args', method='text')
  src.save_data(recordings, location, 'recordings', method='pickle')
  src.save_data(loss_rec, location, 'loss_rec', method='pickle')


