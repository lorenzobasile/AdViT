import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torchvision
from torchvision import transforms
from models import create_ViT
from utils.data import get_dataloaders
from utils.train import train

models = ['mixer_b16_224', 'mixer_l16_224']

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


dataloaders = get_dataloaders(data_dir='./data/imagenette2-320/',
                              train_batch_size=128,
                              test_batch_size=128,
                              data_transforms=data_transforms)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



n_epochs=5
for mod in models:
    print(mod)
    model=timm.create_model(mod, pretrained=True, num_classes=10)
    model=model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)
    for p in model.named_parameters():
        p[1].requires_grad=False
        if p[0]=='head.weight' or p[0]=='head.bias':
            p[1].requires_grad=True
    train(model, dataloaders, n_epochs, optimizer, outfile_name="./training_outputs/"+mod[:-4]+".txt", clip=True)
    torch.save(model.head.state_dict(), "./trained_models/"+mod[:-4]+".pt") #to save memory
