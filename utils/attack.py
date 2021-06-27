import torch


def kpixel_attack(X, X_pgd, k=3000):

    attack_norm = torch.norm(X_pgd - X, dim=1)
    B, C, H, W = X.shape
    reshaped_norm = attack_norm.reshape(B, -1)
    topk = torch.topk(reshaped_norm, k=H * W - k, dim=1, largest=False)
    topkindices = topk.indices[:, :H * W - k]
    for i in range(B):
        reshaped_norm[i][topkindices[i]] = 0
    mask = torch.gt(reshaped_norm, 0).reshape(B, H, W)
    mask = torch.transpose(torch.stack([mask for _ in range(C)]), 0, 1)
    perturbation = torch.mul(X_pgd - X, mask)
    return X + perturbation


def mean_distance(perturbation, p=2):
    B, C, H, W = perturbation.shape
    norm = torch.norm(perturbation, dim=1)
    nz = torch.stack([norm[i].nonzero() for i in range(B)])
    distances = torch.cdist(nz.float(), nz.float(), p=2).reshape(B, -1)
    return distances[distances > 0].mean()
