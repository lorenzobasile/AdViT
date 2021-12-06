import torch

def multidim_select(tensor, indices):
    output_tensor=torch.zeros((tensor.shape[0], indices.shape[1]))
    for i in range(tensor.shape[0]):
        output_tensor[i]=tensor[i,indices[i]]
    return output_tensor

def extract_neighbourhoods(model, n_representations, dataloader, k=10):
    N=len(dataloader.dataset)
    for i, (x,y) in enumerate(train_loader):
        print("Batch: ", i)
        _, repr=model(x)
        if i==0:
            k_dist=torch.ones((len(repr), N, k))*torch.inf
            k_neighbours=torch.zeros((len(repr), N, k), dtype=torch.int)    
        for j, (xp, yp) in enumerate(train_loader):
            _, reprp=model(xp)
            for h in range(n_representations):
                distances, neighbours=[a[:,:k] for a in torch.cdist(representations[h], representationsp[h]).sort(dim=1)]
                neighbours+=j*x.shape[0]
                extended_dist=torch.cat([k_dist[h,i*x.shape[0]:(i+1)*x.shape[0]], distances], axis=1)
                indices=extended_dist.argsort(dim=1)[:,:k]
                k_dist[h, i*x.shape[0]:(i+1)*x.shape[0]]=multidim_select(extended_dist,indices)
                k_neighbours[h, i*x.shape[0]:(i+1)*x.shape[0]]=multidim_select(torch.cat([k_neighbours[h,i*x.shape[0]:(i+1)*x.shape[0]], neighbours], axis=1), indices)
    return k_neighbours
