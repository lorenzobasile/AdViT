import torch

from utils.model import get_model_names
from utils.neighbours import compute_overlap

overlap_model_dict = {}

for model_name in get_model_names():
    k_neighbours = torch.load(f'./neighbourhoods/{model_name}_neighbourhoods.pt') # type torch.Tensor shape (n_repr, N, K)
    chi_l_out = []
    chi_l_l1 = []

    for l in range(k_neighbours.shape[0]-1):
        chi_l_out.append(compute_overlap(k_neighbours[l], k_neighbours[-1]))
        chi_l_l1.append(compute_overlap(k_neighbours[l], k_neighbours[l+1]))

    overlap_model_dict[model_name] = {'chi_l_out': chi_l_out, 'chi_l_l1': chi_l_l1}