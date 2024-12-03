import numpy as np
import math
from scipy.spatial.distance import cdist
from FID import calculate_frechet_distance
import torch
import torch.nn.functional as F
from scipy.io import savemat
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']


import scipy.linalg as linalg

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

codebook_size = 8192
feature_size = 100000
embed_dim = 8
sigma_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
# variance_list = [0.1, 0.5, 1.0, 2.0]
variance_list = [sigma_list[i]*sigma_list[i] for i in range(len(sigma_list))]
repeat_times = 5

quantization_error_list = [[] for i in range(len(variance_list))]
codebook_utilization_list = [[] for i in range(len(variance_list))]
perplexity_list = [[] for i in range(len(variance_list))]

for k in range(repeat_times):

    for i in range(len(variance_list)):
        variance = variance_list[i]

        Gaussian = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(embed_dim), variance * torch.eye(embed_dim))
        
        feature = Gaussian.sample([feature_size])
        feature_mean = feature.mean(0)
        feature_covariance = torch.mm((feature - torch.mean(feature, dim=0, keepdim=True)).t(), feature - torch.mean(feature, dim=0, keepdim=True))/feature_size            
        
        codebook = Gaussian.sample([codebook_size])
        codebook_mean = codebook.mean(0)
        codebook_covariance = torch.mm((codebook - torch.mean(codebook, dim=0, keepdim=True)).t(), codebook - torch.mean(codebook, dim=0, keepdim=True))/codebook_size

        wasserstein_distance = calculate_frechet_distance(feature_mean.cpu().numpy(), feature_covariance.cpu().numpy(), codebook_mean.cpu().numpy(), codebook_covariance.cpu().numpy())

        dist = torch.sum(feature.square(), dim=1, keepdim=True) + torch.sum(codebook.square(), dim=1, keepdim=False)
        dist.addmm_(feature, codebook.T, alpha=-2, beta=1)
        idx = torch.argmin(dist, dim=1)

        nearest_distances = dist.gather(1, idx.unsqueeze(1)).squeeze(1)
        avg_nearest_distance = nearest_distances.mean().item()
        quantization_error_list[i].append(avg_nearest_distance)

        codebook_histogram = idx.bincount(minlength=codebook_size).float()
        codebook_usage_counts = (codebook_histogram > 0).float().sum()

        codebook_utilization = codebook_usage_counts.item() / codebook_size
        codebook_utilization_list[i].append(codebook_utilization)

        #onehot_probs = F.one_hot(idx, codebook_size).type(feature.dtype)
        avg_probs = codebook_histogram/feature_size

        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()
        perplexity_list[i].append(perplexity)
        
        print(avg_nearest_distance)

for i in range(len(variance_list)):
    mean_quantization_error = np.mean(quantization_error_list[i])
    print(f'Gaussain (0, sigma^2): sigma = {sigma_list[i]}, Quantization Error = {mean_quantization_error}')
