#!/usr/bin/env python
# -*- coding: utf-8 -*-



""" 

This script contains the siamese net used for the prediction and 
the class used to compute the Sinkhorn Divergence

Code reference:
https://dfdazac.github.io/sinkhorn.html
"""



#=========================================================================================================
#================================ 0. MODULE



import torch
import torch.nn as nn

import numpy as np


#=========================================================================================================
#================================ 1. SIAMESE NET



class SiameseNet(nn.Module):

    def __init__(self, hidden_layer):
        super(SiameseNet, self).__init__()

        self.feature_extractor = nn.Sequential( nn.Conv2d(1, 16, kernel_size=3),
                                                nn.Conv2d(16, 16, kernel_size=3),
                                                nn.MaxPool2d(2, 2),
                                                nn.Conv2d(16, 32, kernel_size=3),
                                                nn.Conv2d(32, 32, kernel_size=3),
                                                nn.MaxPool2d(2, 2),
                                                nn.ReLU())

        self.conv_out_size = self._get_conv_out((1, 28, 28))

        self.embedding = nn.Sequential( nn.Linear(self.conv_out_size, hidden_layer),
                                        nn.ReLU(),
                                        nn.Linear(hidden_layer, 64) )


    def _get_conv_out(self, shape):
        """
        Incredibly smart function used to automatically compute
        the size of the convolution output layer
        """
        o = self.feature_extractor(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(-1, self.conv_out_size)
        out = self.embedding(out)
        return out



#=========================================================================================================
#================================ 2. SINKHORN DIVERGENCE



class SinkhornDivergenceLoss(nn.Module):
    """
    Compute the Sinkhorn Divergence between two distributions seen as two point clouds.

    Arguments:
    ----------
    lbda (float): regularization coefficient
    max_iter (int): maximum number of Sinkhorn iterations
    reduction (string, optional): None or mean

    """
    def __init__(self, lbda, max_iter, p=2, reduction=None):
        super(SinkhornDivergenceLoss, self).__init__()

        self.lbda = lbda
        self.max_iter = max_iter
        self.reduction = reduction
        self.p = p


    def forward(self, distrib_x, distrib_y):
        """
        Arguments:
        ----------
        distrib: tensor of shape [batch_size, support_points, dimension]

        Return:
        loss: mean of all SinkhornDivergence (except if reduction is 'none')
        """

        # Device
        device = distrib_x.device

        # Compute distance matrix 
        x_cols = distrib_x.unsqueeze(-2)
        y_lines = distrib_y.unsqueeze(-3)
        D = torch.sum((torch.abs(x_cols - y_lines)) ** self.p, -1)
        D = D.to(device)

        # Point cloud
        x_points = distrib_x.shape[-2]
        y_points = distrib_y.shape[-2]

        # Batch size
        if distrib_x.dim() == 2:
            batch_size = 1
        else:
            batch_size = distrib_x.shape[0]

        # Uniform discrete distributions
        u = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
        v = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)


        # Stopping criterion
        threshold = 1e-2

        # Initial value
        K = -D
        r = torch.zeros_like(u).to(device)
        c = torch.zeros_like(v).to(device)

        # Sinkhorn iterations (using log for better numerical stability)
        for i in range(self.max_iter):
            previous_u = u.clone()

            # Update r and c
            r += self.lbda * (torch.log(u + 1e-10) 
                                    - torch.logsumexp( (r.unsqueeze(-1) + K + c.unsqueeze(-2)) / self.lbda, dim=-1) )
            c += self.lbda * (torch.log(v + 1e-10) 
                                    - torch.logsumexp( ((r.unsqueeze(-1) + K + c.unsqueeze(-2))  / self.lbda ).transpose(-2, -1), dim=-1) )

            diff = (u - previous_u).abs().sum(-1).mean()

            if diff.item() < threshold:
                break


        # Optimal T
        T = torch.exp( (r.unsqueeze(-1) + K + c.unsqueeze(-2)) / self.lbda )

        # Sinkhorn divergence
        loss = torch.sum(T * D, dim=(-2, -1))

        if self.reduction is not None:
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss