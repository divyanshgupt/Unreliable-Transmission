def poisson_trains(n, lam, args):
  """

  inputs:
    n - number of poisson spike trains 
    lam - 1-D array containing mean value of poisson trains
  Returns

  """
  timesteps = args['nb_steps']
  dt = args['timestep_size']
  
  trains = torch.zeros((n, timesteps), device=device, dtype=dtype)
  unif = torch.rand((n, timesteps), device=device, dtype=dtype)

#  counter = 0
  for i in range(n):
    trains[unif <= lam[i]*dt] = 1
#    counter += len(unif <= lam[i]*dt)
#  print("Total No. of Spikes", counter)

  return trains