import timm
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms, models
from utils import WarmupCosineSchedule

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10, img_size=32)
device="cuda:0"
model = model.to(device)

trainset = torchvision.datasets.CIFAR10("./data", train=True, transform=transforms.ToTensor(), download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10("./data", train=False, transform=transforms.ToTensor(), download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, weight_decay=args.weight_decay)
loss = torch.nn.CrossEntropyLoss()
scheduler=WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=1000)

for epoch in range(1000):
    print("Epoch: ", epoch, '/', 1000)
    model.train()
    scheduler.step()
    for x, y in trainloader:
        x=x.to(device)
        y=y.to(device)
        out=model(x)
        l=loss(out, y)
        optimizer.zero_grad()
        l.backward()
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
