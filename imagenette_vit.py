import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import torchvision
from torchvision import transforms
from models import create_ViT
from utils import train

#model = create_ViT(img_size=224, patch_size=16, num_classes=10)

model=timm.create_model('tv_resnet101', num_classes=10)

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


data_dir = './data/imagewoof2-320/'
datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {'train': DataLoader(datasets['train'], batch_size=128, shuffle=True),'test': DataLoader(datasets['test'], batch_size=128, shuffle=False)}
#dataset_sizes = {x: len(datasets[x]) for x in ['Training', 'Testing']}
#class_names = datasets['Training'].classes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, weight_decay=0)
loss = torch.nn.CrossEntropyLoss()
#scheduler=WarmupCosineSchedule(optimizer, warmup_steps=100, t_total=200)

n_epochs=5

for p in model.named_parameters():
    p[1].requires_grad=False
    if p[0]=='fc.weight' or p[0]=='fc.bias':
        p[1].requires_grad=True

train(model, dataloaders, n_epochs, optimizer, outfile_name="imagewoof_vitb16.txt", clip=True)
'''
for epoch in range(n_epochs):
    with open("imagenette_vit.txt", 'a') as outfile:
            outfile.write("\nEpoch: "+str(epoch)+'/'+str(n_epochs))
    print("Epoch: ", epoch, '/', n_epochs)
    model.train()
    #scheduler.step()
    for x, y in dataloaders['train']:
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
        for x, y in dataloaders['train']:
            out=model(x.to(device))
            correct+=(torch.argmax(out, axis=1)==y.to(device)).sum().item()
    with open("imagenette_vit.txt", 'a') as outfile:
             outfile.write("\nAccuracy on train set: "+str(correct/len(datasets['train'])))
    print("Accuracy on training set: ", correct/len(datasets['train']))

    correct=0
    with torch.no_grad():
        for x, y in dataloaders['test']:
            out=model(x.to(device))
            correct+=(torch.argmax(out, axis=1)==y.to(device)).sum().item()
    with open("imagenette_vit.txt", 'a') as outfile:
        outfile.write("\nAccuracy on test set: "+str(correct/len(datasets['test'])))
    print("Accuracy on test set: ", correct/len(datasets['test']))
#torch.save(model.state_dict(), "./trained_models/vit_imagenette.pt")
'''
torch.save(model.head.state_dict(), "./trained_models/vitb16_imagewoof_head.pt") #to save memory
