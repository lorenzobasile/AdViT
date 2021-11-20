# TODO: remove this file?

import timm
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from radam import RAdam

from utils.model import create_ViT
from utils.scheduler import WarmupCosineSchedule
from utils.train import train

model = create_ViT(img_size=32, patch_size=4, num_classes=10)
device="cuda:0"

model=model.to(device)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

trainset = torchvision.datasets.CIFAR10("./data", train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testset = torchvision.datasets.CIFAR10("./data", train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

optimizer = RAdam(model.parameters(), lr=0.03)
loss = torch.nn.CrossEntropyLoss()
scheduler=WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=1000)

dataloaders={'train':trainloader, 'test':testloader}

train(model, dataloaders, 1000, optimizer, scheduler, outfile_name="./training_outputs/cifar_radam_vit.txt")

torch.save(model.state_dict(), "./trained_models/cifar_radam_vit.pt")
