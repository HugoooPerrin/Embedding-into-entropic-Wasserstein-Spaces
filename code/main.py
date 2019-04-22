#!/usr/bin/env python
# -*- coding: utf-8 -*-



""" 

Main script to learn embeddings into Entropic Wasserstein Spaces


"""



#=========================================================================================================
#================================ 0. MODULE


# General
import numpy as np


# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


# Personal libraries
from model import SiameseNet, SinkhornDivergenceLoss
from data import ImageTriplet


# Utils
from datetime import datetime
from dateutil.relativedelta import relativedelta
def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


#=========================================================================================================
#================================ 1. LOADING DATA


EPOCHS = 20
BATCH_SIZE = 1024

display_step = 100

loader = DataLoader(dataset=ImageTriplet(data='full'),
                    shuffle=True,
                    batch_size=BATCH_SIZE,
                    num_workers=8)


#=========================================================================================================
#================================ 2. BUILDING MODEL


device = 'cuda'

# Hyperparameters
LEARNING_RATE = 1e-3
HIDDEN_LAYER = 100
M = 10

# Model
model = SiameseNet(100).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = SinkhornDivergenceLoss(lbda=0.1, max_iter=100, p=2, reduction='none')

# Count the number of parameters in the network
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('\n>> Learning: {} parameters\n'.format(params))


#=========================================================================================================
#================================ 3. MAIN

D = 4

# Training procedure
time = datetime.now()
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.
    i = 1

    for batch in loader:

        image, positive_sample, negative_sample = batch

        image = image.to(device)
        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)

        # Computing embedding
        image = model(image)
        positive_sample = model(positive_sample)
        negative_sample = model(negative_sample)
        
        if D == 2:
            image = image.reshape(-1, 32, 2)
            positive_sample = positive_sample.reshape(-1, 32, 2)
            negative_sample = negative_sample.reshape(-1, 32, 2)

        elif D == 4:
            image = image.reshape(-1, 16, 4)
            positive_sample = positive_sample.reshape(-1, 16, 4)
            negative_sample = negative_sample.reshape(-1, 16, 4)

        # Sinkhorn Divergence for same label images
        same_loss = criterion(image, positive_sample)

        # Sinkhorn Divergence for different label images
        diff_loss = criterion(image, negative_sample)

        # Contrastive loss function
        loss = (same_loss ** 2).mean() + (torch.max(M - diff_loss, torch.zeros_like(diff_loss)) ** 2).mean()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Compute gradient using backpropagation
        loss.backward()

        # Take a optimizer step
        optimizer.step()

        # Save batch loss
        epoch_loss += loss.data.item()

        # Monitoring performance
        if i % display_step == 0:

            print('Epoch: %2d, step: %4d, mean sinkhorn divergence: %.4f' %
                  (epoch, i, epoch_loss / i))

        i += 1

    # break

    # Saving model
    print('\nRunning time: {}\n'.format(diff(datetime.now(), time)))
    torch.save(model.state_dict(), "../models/point_cloud_embedding.model")