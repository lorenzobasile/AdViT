import os

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_dataloaders(data_dir, train_batch_size, test_batch_size, data_transforms=None, shuffle_train=True, shuffle_test=False):
    if data_transforms is None:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ]),
        }

    datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                ['train', 'test']}
    dataloaders = {'train': DataLoader(datasets['train'], batch_size=train_batch_size, shuffle=shuffle_train),
                   'test': DataLoader(datasets['test'], batch_size=test_batch_size, shuffle=shuffle_test)}
    return dataloaders
