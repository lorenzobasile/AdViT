import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torchvision
from torchvision import transforms
from models import create_ViT
from utils import train

vit_models = ['vit_base_patch16_224_in21k', 'vit_large_patch16_224_in21k', 'vit_base_patch32_224_in21k',  'vit_large_patch32_224_in21k']

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


data_dir = './data/imagenette2-320/'
datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {'train': DataLoader(datasets['train'], batch_size=128, shuffle=True),'test': DataLoader(datasets['test'], batch_size=128, shuffle=False)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



n_epochs=5
for vit in vit_models:
    print(vit)
    model=timm.create_model(vit, pretrained=True, num_classes=10)
    model=model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)
    for p in model.named_parameters():
        p[1].requires_grad=False
        if p[0]=='head.weight' or p[0]=='head.bias':
            p[1].requires_grad=True
    train(model, dataloaders, n_epochs, optimizer, outfile_name="./training_outputs/"+vit[4:-10]+".txt", clip=True)
    torch.save(model.head.state_dict(), "./trained_models/"+vit[4:-10]+".pt") #to save memory
