import timm
import torch
from torch.utils.data import DataLoader
from deeprobust.image.attack.fgsm import FGSM
from torchvision import transforms
import torchvision
import os

outfile_name="./attack_results/fgsm.txt"

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
dataloaders = {'train': DataLoader(datasets['train'], batch_size=128, shuffle=True),'test': DataLoader(datasets['test'], batch_size=64, shuffle=False)}

model_names=['resnet18', 'tv_resnet50', 'tv_resnet101', 'vgg16', 'vit_base_patch16_224',  'vit_base_patch32_224',  'vit_small_patch16_224','vit_small_patch32_224']
epsilons=[0.001, 0.005, 0.01, 0.05, 0.1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models=[timm.create_model(model_name, pretrained=True, num_classes=10).to(device) for model_name in model_names]


for i, model_name in enumerate(model_names):
    print(model_name)
    if 'vit' in model_name:
        models[i].head.load_state_dict(torch.load("./trained_models/in1k"+model_name[4:-4]+".pt"))
    elif 'vgg' in model_name:
        models[i].head.fc.load_state_dict(torch.load("./trained_models/vgg16.pt"))
    else:
        models[i].fc.load_state_dict(torch.load("./trained_models/"+model_name+".pt"))
    models[i].eval()
correct=0

correct=torch.zeros(len(models))
for x, y in dataloaders['test']:
        x=x.to(device)
        y=y.to(device)
        for k, model in enumerate(models):
            temp=torch.argmax(model(x), axis=1)==y
            correct[k]+=temp.sum().item()
with open(outfile_name, 'w') as outfile:
    outfile.write("Clean accuracy: ", correct/len(dataloaders['test'].dataset))
print("Clean accuracy: ", correct/len(dataloaders['test'].dataset))

adversaries=[FGSM(model, 'cuda') for model in models]
for eps in epsilons:
    for i, attacked_model in enumerate(models):
        correct=torch.zeros(len(models))
        for x, y in dataloaders['test']:
            x=x.to(device)
            y=y.to(device)
            x=adversaries[i].generate(x, y, epsilon=eps)
            for k, model in enumerate(models):
                temp=torch.argmax(model(x), axis=1)==y
                correct[k]+=temp.sum().item()
        with open(outfile_name, 'a') as outfile:
             outfile.write("\nAttack on "+model_names[i]+": ", correct/len(dataloaders['test'].dataset))
        print("Attack on "+model_names[i]+": ", correct/len(dataloaders['test'].dataset))
