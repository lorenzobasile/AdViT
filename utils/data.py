import os

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_dataloaders(data_dir, train_batch_size, test_batch_size, data_transforms=None, shuffle_train=True, shuffle_test=False):
    """
    Get dataloaders for training and testing on data stored in a directory.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the data.
    train_batch_size : int
        Batch size for training.
    test_batch_size : int
        Batch size for testing.
    data_transforms : torchvision.transforms.Compose
        Transformation to apply to the data.
    shuffle_train : bool
        Whether to shuffle the training data.
    shuffle_test : bool
        Whether to shuffle the testing data.

    Returns
    -------
    data_loaders : dict
        Dictionary containing the dataloaders for training and testing.
    """
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
