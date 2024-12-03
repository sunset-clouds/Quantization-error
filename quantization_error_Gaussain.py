import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

codebook_size = 8192
feature_size = 100000
embed_dim = 8
sigma_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
variance_list = [sigma_list[i]*sigma_list[i] for i in range(len(sigma_list))]
repeat_times = 5

quantization_error_list = [[] for i in range(len(variance_list))]

for k in range(repeat_times):

    for i in range(len(variance_list)):
        variance = variance_list[i]
        Gaussian = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(embed_dim), variance * torch.eye(embed_dim))
        
        feature = Gaussian.sample([feature_size])      
        codebook = Gaussian.sample([codebook_size])

        dist = torch.sum(feature.square(), dim=1, keepdim=True) + torch.sum(codebook.square(), dim=1, keepdim=False)
        dist.addmm_(feature, codebook.T, alpha=-2, beta=1)
        idx = torch.argmin(dist, dim=1)
        nearest_distances = dist.gather(1, idx.unsqueeze(1)).squeeze(1)
        avg_nearest_distance = nearest_distances.mean().item()
        quantization_error_list[i].append(avg_nearest_distance)

for i in range(len(variance_list)):
    mean_quantization_error = np.mean(quantization_error_list[i])
    print(f'Gaussain (0, sigma^2): sigma = {sigma_list[i]}, Quantization Error = {mean_quantization_error}')