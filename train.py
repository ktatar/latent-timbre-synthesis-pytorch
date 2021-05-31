# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import random
import numpy as np

import os, sys, argparse, time
from pathlib import Path

import librosa
import configparser
import random
import json
import matplotlib.pyplot as plt
import pdb

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
args = parser.parse_args()

#Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
try: 
  config.read(config_path)
except FileNotFoundError:
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()

#import audio configs 
sample_rate = config['audio'].getint('sample_rate')
hop_length = config['audio'].getint('hop_length')
bins_per_octave = config['audio'].getint('bins_per_octave')
num_octaves = config['audio'].getint('num_octaves')
n_bins = int(num_octaves * bins_per_octave)
n_iter = config['audio'].getint('n_iter')

#dataset
dataset = Path(config['dataset'].get('datapath'))
if not dataset.exists():
    raise FileNotFoundError(dataset.resolve())

cqt_dataset = config['dataset'].get('cqt_dataset')

if config['dataset'].get('workspace') != None:
  workspace = Path(config['dataset'].get('workspace'))

run_number = config['dataset'].getint('run_number')
my_cqt = dataset / cqt_dataset
if not my_cqt.exists():
    raise FileNotFoundError(my_cqt.resolve())

my_audio = dataset / 'audio'
    
#Training configs
epochs = config['training'].getint('epochs')
learning_rate = config['training'].getfloat('learning_rate')
batch_size = config['training'].getint('batch_size')

#Model configs
latent_dim = config['VAE'].getint('latent_dim')
n_units = config['VAE'].getint('n_units')
kl_beta = config['VAE'].getfloat('kl_beta')
device = config['VAE'].get('device')

#etc
example_length = config['extra'].getint('example_length')
normalize_examples = config['extra'].getboolean('normalize_examples')
plot_model = config['extra'].getboolean('plot_model')

desc = config['extra'].get('description')
start_time = time.time()
config['extra']['start'] = time.asctime( time.localtime(start_time) )

device = torch.device(device)
if torch.cuda.is_available():
  print(torch.cuda.get_device_name())

else:
  print("Running on CPU.......")

#Create workspace


run_id = run_number
while True:
    try:
        my_runs = dataset / desc
        run_name = 'run-{:03d}'.format(run_id)
        workdir = my_runs / run_name 
        os.makedirs(workdir)

        break
    except OSError:
        if workdir.is_dir():
            run_id = run_id + 1
            continue
        raise

config['dataset']['workspace'] = str(workdir.resolve())

print("Workspace: {}".format(workdir))

#create the dataset
print('creating the dataset...')
training_array = []
new_loop = True

for f in os.listdir(my_cqt): 
    if f.endswith('.npy'):
        print('adding-> %s' % f)
        file_path = my_cqt / f
        new_array = np.load(file_path)
        if new_loop:
            training_array = new_array
            new_loop = False
        else:
            training_array = np.concatenate((training_array, new_array), axis=0)

total_cqt = len(training_array)
print('Total number of CQT frames: {}'.format(total_cqt))
config['dataset']['total_frames'] = str(total_cqt)

training_tensor = torch.Tensor(training_array)
training_dataset = TensorDataset(training_tensor)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

print("saving initial configs...")
config_path = workdir / 'config.ini'
with open(config_path, 'w') as configfile:
  config.write(configfile)

# Train
model_dir = workdir / "model"
os.makedirs(model_dir,exist_ok=True)

log_dir = workdir / 'logs'
os.makedirs(log_dir, exist_ok=True)

class VAE(nn.Module):
  def __init__(self, n_bins, n_units, latent_dim):
    super(VAE, self).__init__()

    self.n_bins = n_bins
    self.n_units = n_units
    self.latent_dim = latent_dim
    
    self.fc1 = nn.Linear(n_bins, n_units)
    self.fc21 = nn.Linear(n_units, latent_dim)
    self.fc22 = nn.Linear(n_units, latent_dim)
    self.fc3 = nn.Linear(latent_dim, n_units)
    self.fc4 = nn.Linear(n_units, n_bins)

  def encode(self, x):
      h1 = F.relu(self.fc1(x))
      return self.fc21(h1), self.fc22(h1)

  def reparameterize(self, mu, logvar):
      std = torch.exp(0.5*logvar)
      eps = torch.randn_like(std)
      return mu + eps*std

  def decode(self, z):
      h3 = F.relu(self.fc3(z))
      return F.relu(self.fc4(h3))

  def forward(self, x):
      mu, logvar = self.encode(x.view(-1, self.n_bins))
      z = self.reparameterize(mu, logvar)
      return self.decode(z), mu, logvar


model = VAE(n_bins, n_units, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, kl_beta):
    recon_loss = F.mse_loss(recon_x, x.view(-1, n_bins))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + ( kl_beta * KLD)

for epoch in range(epochs):
  print('Epoch {}/{}'.format(epoch, epochs - 1))
  print('-' * 10)

  model.train()
  train_loss = 0
  for i, data in enumerate(training_dataloader):
    data, = data
    data = data.to(device)
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    loss = loss_function(recon_batch, data, mu, logvar, kl_beta)
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
  print('====> Epoch: {} Average loss: {:.8f}'.format(
          epoch, train_loss / len(training_dataloader.dataset)))

pdb.set_trace()

save_path = workdir.joinpath('model').joinpath('last_model.pt')
torch.save(model, save_path)
