import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torchvision
from torchvision import transforms
from models import create_ViT
from utils import train

cnn_names = ['resnet18', 'tv_resnet34','tv_resnet50', 'tv_resnet101', 'wide_resnet50_2', 'wide_resnet101_2', 'vgg16']

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
for cnn in cnn_names:
    print(cnn)
    model=timm.create_model(cnn, pretrained=True, num_classes=10)
    model=model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)
    for p in model.named_parameters():
        p[1].requires_grad=False
        if p[0]=='fc.weight' or p[0]=='head.fc.weight' or p[0]=='fc.bias' or p[0]=='head.fc.bias':
            p[1].requires_grad=True
    train(model, dataloaders, n_epochs, optimizer, outfile_name=cnn + ".txt", clip=True)
    torch.save(model.head.state_dict(), cnn + ".pt") #to save memory
