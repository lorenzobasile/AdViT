import timm
import torch
from torch.utils.data import DataLoader
from deeprobust.image.attack.pgd import PGD
from torchvision import transforms
import torchvision
import os
from utils.attack import kpixel_attack
from utils.data import get_dataloaders

outfile_name="./attack_results/pgd_robust.txt"

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ]),
}


dataloaders = get_dataloaders(data_dir='./data/imagenette2-320/',
                              train_batch_size=128,
                              test_batch_size=64,
                              data_transforms=data_transforms)

model_names=['tv_resnet101','vit_base_patch16_224']
epsilons=[0.0005, 0.001, 0.005, 0.01]
#epsilons=[0.1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models=[timm.create_model(model_name, pretrained=True, num_classes=10).to(device) for model_name in model_names]


for i, model_name in enumerate(model_names):
    if 'vit' in model_name:
        models[i].load_state_dict(torch.load("./trained_models/"+model_name+"_PGD_eps0.0100.pt"))
    else:
        models[i].load_state_dict(torch.load("./trained_models/"+model_name+"_PGD_eps0.0100.pt"))
    models[i].eval()

#correct=0

correct=torch.zeros(len(models))
for x, y in dataloaders['test']:
        x=x.to(device)
        y=y.to(device)
        for k, model in enumerate(models):
            temp=torch.argmax(model(x), axis=1)==y
            correct[k]+=temp.sum().item()
with open(outfile_name, 'w') as outfile:
    outfile.write("Clean accuracy: "+str(correct/len(dataloaders['test'].dataset)))
print("Clean accuracy: ", correct/len(dataloaders['test'].dataset))
adversaries=[PGD(model, 'cuda') for model in models]
for eps in epsilons:
    with open(outfile_name, 'a') as outfile:
        outfile.write("\nEpsilon: "+str(eps))
    for i, attacked_model in enumerate(models):
        correct=torch.zeros(len(models))
        #correct_k=torch.zeros(len(models))
        for j, (x, y) in enumerate(dataloaders['test']):
            x=x.to(device)
            y=y.to(device)
            perturbed_x=adversaries[i].generate(x, y, epsilon=eps, step_size=eps/3, num_steps=10)
            for k, model in enumerate(models):
                temp=torch.argmax(model(perturbed_x), axis=1)==y
                correct[k]+=temp.sum().item()
            #perturbed_x=kpixel_attack(x, perturbed_x, k=3000)
            '''
            for k, model in enumerate(models):
                temp=torch.argmax(model(perturbed_x), axis=1)==y
                correct_k[k]+=temp.sum().item()
            '''
        with open(outfile_name, 'a') as outfile:
             outfile.write("\nAttack on "+model_names[i]+": "+str(correct/len(dataloaders['test'].dataset)))
             #outfile.write("\nk-Attack on "+model_names[i]+": "+str(correct_k/len(dataloaders['test'].dataset)))
        print("Attack on "+model_names[i]+": ", correct/len(dataloaders['test'].dataset))
