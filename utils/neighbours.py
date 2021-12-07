import torch
import numpy as np

def multidim_select(tensor, indices):
    output_tensor=torch.zeros((tensor.shape[0], indices.shape[1]))
    for i in range(tensor.shape[0]):
        output_tensor[i]=tensor[i,indices[i]]
    return output_tensor

def extract_neighbourhoods(model, dataloader, k=10):
    with torch.no_grad():
        device=torch.device('cuda:1')
        N=len(dataloader.dataset)
        for i, (x,y) in enumerate(dataloader):
            x=x.to(device)
            print("Batch: ", i)
            _, repr=model(x)
            if i==0:
                k_dist=torch.ones((len(repr), N, k)).to(device)*np.inf
                k_neighbours=torch.zeros((len(repr), N, k), dtype=torch.int).to(device)
            for j, (xp, yp) in enumerate(dataloader):
                torch.cuda.empty_cache()
                xp=xp.to(device)
                _, reprp=model(xp)
                for h in range(len(repr)):
                    distances, neighbours=[a[:,:k] for a in torch.cdist(repr[h].reshape(repr[h].shape[0], -1), reprp[h].reshape(reprp[h].shape[0], -1)).sort(dim=1)]
                    neighbours=neighbours.int()
                    neighbours+=j*x.shape[0]
                    extended_dist=torch.cat([k_dist[h,i*x.shape[0]:(i+1)*x.shape[0]], distances], axis=1)
                    indices=extended_dist.argsort(dim=1)[:,:k]
                    k_dist[h, i*x.shape[0]:(i+1)*x.shape[0]]=multidim_select(extended_dist,indices)
                    k_neighbours[h, i*x.shape[0]:(i+1)*x.shape[0]]=multidim_select(torch.cat([k_neighbours[h,i*x.shape[0]:(i+1)*x.shape[0]], neighbours], axis=1), indices)
    return k_neighbours
