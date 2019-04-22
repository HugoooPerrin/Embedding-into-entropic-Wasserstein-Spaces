#!/usr/bin/env python
# -*- coding: utf-8 -*-



""" 

Preprocessing of the data so as to feed the algorithm


"""



#=========================================================================================================
#================================ 0. MODULE



import torch
from torch.utils.data import Dataset

import numpy as np
from imageio import imread

import random

from math import floor

from tqdm import tqdm

import os
import pickle


#=========================================================================================================
#================================ 1. DATASET CLASS 


PATH = '/home/hugoperrin/Bureau/Datasets/notMNIST/notMNIST_large/notMNIST_large/'
DATA = '/home/hugoperrin/Bureau/ENSAE/3A/Cours/Geometric methods for machine learning/Projet'


LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
N = 40000


class ImageTriplet(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data='full'):
      
        # Opening data
        if data == 'full':
            self.data_dict = pickle.load(open(DATA + '/data.p', 'rb'))
            self.N = 40000
        elif data == 'small':
            self.data_dict = pickle.load(open(DATA + '/data_small.p', 'rb'))
            self.N = 1000

        # Total number of datapoints
        self.dataset_size = self.data_dict['A'].shape[0] * 10


    def __getitem__(self, i):

        # Get concerned letter
        L_idx = floor(i / self.N)
        L = LETTERS[L_idx]

        # Get image index
        image_idx = i % self.N
        
        # Image
        image = torch.FloatTensor(self.data_dict[L][image_idx]).unsqueeze(0)

        # Positive sample
        random_i = np.random.randint(self.N)
        positive_sample = torch.FloatTensor(self.data_dict[L][random_i]).unsqueeze(0)

        # Negative sample
        random_L = random.choice([letter for letter in LETTERS if letter != L])
        random_j = np.random.randint(self.N)
        negative_sample = torch.FloatTensor(self.data_dict[random_L][random_j]).unsqueeze(0)

        return image, positive_sample, negative_sample


    def __len__(self):
        return self.dataset_size



#=========================================================================================================
#================================ 2. MAIN


if __name__ == '__main__':

    """
    Open N images for each letter, normalize them and save all 
    as a dictionary.
    """

    data = {}

    for L in LETTERS:
        print('\nOpening {} images'.format(L))
        image_paths = os.listdir(PATH + L)
        data[L] = []
        
        for p in tqdm(image_paths):
            try:
                # Opening image and normalizing
                data[L].append(imread(PATH + L + '/' + p)[np.newaxis, :, :] / 255.)
            except:
                pass

            if len(data[L]) == N:
                break
        
        data[L] = np.concatenate(data[L])

    # Saving data
    pickle.dump(data, open(DATA + '/data_small.p', 'wb'))

