import timm
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms, models
from utils import WarmupCosineSchedule

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10, img_size=32)
device="cuda:0"

for p in model.named_parameters():
    if p[0]=='patch_embed.proj.bias':
        biases=p[1]
    if p[0]=='patch_embed.proj.weight':
        weights=p[1]

sampled_weights=weights[:,:,::4,::4]

model.patch_embed.proj=nn.Conv2d(3,768,(4,4),(4,4))

model.patch_embed.proj.weight=torch.nn.Parameter(sampled_weights)
model.patch_embed.proj.bias=torch.nn.Parameter(biases)

model.patch_embed.num_patches=64

model.pos_embed = nn.Parameter(torch.zeros(1, 64 + 1, 768))

model=model.to(device)

transform=transforms.Compose([
    #transforms.Resize((96, 96)),
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
torch.save(model.state_dict(), "./models/vit_sgd.pt")
