import timm
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from utils import WarmupCosineSchedule
from models import create_ViT

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

#optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, weight_decay=0)
loss = torch.nn.CrossEntropyLoss()
scheduler=WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=1000)

for epoch in range(500):
    print("Epoch: ", epoch, '/', 500)
    model.train()
    scheduler.step()
    for x, y in trainloader:
        x=x.to(device)
        y=y.to(device)
        out=model(x)
        l=loss(out, y)
        optimizer.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    correct=0
    with torch.no_grad():
        for x, y in trainloader:
            out=model(x.to(device))
            correct+=(torch.argmax(out, axis=1)==y.to(device)).sum().item()
    print("Accuracy on training set: ", correct/len(trainset))

    correct=0
    with torch.no_grad():
        for x, y in testloader:
            out=model(x.to(device))
            correct+=(torch.argmax(out, axis=1)==y.to(device)).sum().item()
    print("Accuracy on test set: ", correct/len(testset))
torch.save(model.state_dict(), "./trained_models/cifar_vit.pt")