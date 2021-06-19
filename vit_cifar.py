import timm
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms, models
from utils import WarmupCosineSchedule

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10, img_size=96)
device="cuda:0"
model = model.to(device)

transform=transforms.Compose([
    #transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

#trainset = torchvision.datasets.CIFAR10("./data", train=True, transform=transform, download=True)
trainset=torchvision.datasets.STL10("./data", 'train', transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testset=torchvision.datasets.STL10("./data", 'test', transform=transform, download=True)

#testset = torchvision.datasets.CIFAR10("./data", train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, weight_decay=0)
loss = torch.nn.CrossEntropyLoss()
scheduler=WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=500)

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
torch.save(model.state_dict(), "./models/vit_sgd.pt")
