import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

codebook_size = 8192
feature_size = 100000
embed_dim = 8
v_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
repeat_times = 5

quantization_error_list = [[] for i in range(len(v_list))]

for k in range(repeat_times):

    for i in range(len(v_list)):
        v = v_list[i]
        Uniform = torch.distributions.uniform.Uniform(-v * torch.ones(embed_dim, device=device), v * torch.ones(embed_dim, device=device))   
        
        feature = Uniform.sample([feature_size]) 
        codebook = Uniform.sample([codebook_size])

        dist = torch.sum(feature.square(), dim=1, keepdim=True) + torch.sum(codebook.square(), dim=1, keepdim=False)
        dist.addmm_(feature, codebook.T, alpha=-2, beta=1)
        idx = torch.argmin(dist, dim=1)
        nearest_distances = dist.gather(1, idx.unsqueeze(1)).squeeze(1)
        avg_nearest_distance = nearest_distances.mean().item()
        quantization_error_list[i].append(avg_nearest_distance)

for i in range(len(v_list)):
    mean_quantization_error = np.mean(quantization_error_list[i])
    print(f'Unif(-v, v): v = {v_list[i]}, Quantization Error = {mean_quantization_error}')
