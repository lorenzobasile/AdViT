import os

import torchvision
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, train_batch_size, test_batch_size, data_transforms, shuffle_train=True, shuffle_test=False):
    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                ['train', 'test']}
    dataloaders = {'train': DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=shuffle_train),
                   'test': DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=shuffle_test)}
    return dataloaders
