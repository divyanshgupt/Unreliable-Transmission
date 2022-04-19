import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from datetime import datetime
import pickle
import src
import os
from src.params_multilayer import *

# set device
dtype = torch.float
cpu = torch.device("cpu")
gpu = torch.device("cuda:0")

device = cpu
# Uncomment the line below to run on GPU
#device = gpu

note = ""

nb_trials = 1

# 100 independent poisson trains
spk_freq = 10
input_trains = src.poisson_trains(nb_inputs, spk_freq*np.ones(nb_inputs), args)

# 5 equidistant spikes over 500 msec.
target = torch.zeros((nb_steps), device=device, dtype=dtype)
target[500::int(nb_steps/5)] = 1

#@title Training the network

#weight_scale = 100

#w1, w2 = src.initialize_weights_multilayer(nb_inputs, nb_hidden, nb_outputs, args, weight_scale)

w1, w2 = src.new_initialize_weights_multilayer(nb_inputs, nb_hidden, nb_outputs, args) #shape: (nb_inputs, nb_hidden), (nb_hidden, nb_outputs)

feedback_type = 'random'

# Random feedback
#feedback_weights = src.random_feedback(nb_hidden, nb_outputs, args).T # shape: (nb_outputs, nb_hidden)

# Uniform Feedback
#feedback_weights = torch.ones((nb_outputs, nb_hidden), device=device, dtype=dtype)

# Symmetric Feedback
feedback_weights = w2.T


gamma = float(np.exp(-dt/args['tau_rms']))

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

  data_folder = "data/multilayer/" + \
        str(datetime.today())[:13] + ' rate = ' + str(r_0) + '/'
  location = os.path.abspath(data_folder)
  location = os.path.join(os.getcwd(), location)
  os.makedirs(location)



  plt.savefig(location + "/loss over epochs" + "learning-rate = " + str(r_0) +
              ", epsilon = " + str(args['epsilon']) + "spike freq = " + str(spk_freq) + ".png")


  loss_file_name = location + "/loss_rec epsilon= " + \
      str(args['epsilon']) + "learning_rate = " + \
      str(r_0) + "spike freq = " + str(spk_freq)
  loss_file = open(loss_file_name, 'wb')
  pickle.dump(loss_rec, loss_file)
  loss_file.close()

  # Store args:
  file_name = location + "/args epsilon = " + \
      str(args['epsilon']) + " learning_rate = " + \
      str(r_0) + " spike freq = " + str(spk_freq)
  args_file = open(file_name, 'a')
  args_file.write(str(args) + '\n \n')
  args_file.write(note)
  args_file.close()

  date_stamp = str(datetime.today())
  location = src.set_location(f'data/multilayer/{feedback_type}/')
  src.save_data(f'{args} \n \n {note}', location, f'args', method='text')
  src.save_data(recordings, location, 'recordings', method='pickle')
  src.save_data(loss_rec, location, 'loss_rec', method='pickle')

  recordings_filename = location + "/recordings epsilon= " + \
        str(args['epsilon']) + "learning_rate = " + \
        str(r_0) + "spike freq = " + str(spk_freq)

  recordings_file = open(recordings_filename, 'wb')
  pickle.dump(recordings_list, recordings_file)
  recordings_file.close()

