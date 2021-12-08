import timm
import torch
from deeprobust.image.attack.pgd import PGD
from utils.data import get_dataloaders
from utils.adversarial.attack import kpixel_attack, mean_weighted_distance
from torchvision import transforms


outfile_name="./attack_results/weighted_dispersion.txt"

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]),
}
dataloaders = get_dataloaders(data_dir='./data/imagenette2-320/',
                              train_batch_size=128,
                              test_batch_size=64,
                              data_transforms=data_transforms)

model_names=['mixer_b16_224','resnet18', 'tv_resnet50', 'tv_resnet101', 'vgg16', 'vit_base_patch16_224',  'vit_base_patch32_224',  'vit_small_patch16_224','vit_small_patch32_224']
epsilons=[0.001, 0.005, 0.01, 0.05, 0.1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models=[timm.create_model(model_name, pretrained=True, num_classes=10).to(device) for model_name in model_names]
kappa=range(1000,2100,100)

for i, model_name in enumerate(model_names):
    if model_name=='mixer_b16_224':
        models[i].head.load_state_dict(torch.load("./trained_models/"+model_name[:-4]+".pt"))
    elif 'vit' in model_name:
        models[i].head.load_state_dict(torch.load("./trained_models/in1k"+model_name[4:-4]+".pt"))
    elif 'vgg' in model_name:
        models[i].head.fc.load_state_dict(torch.load("./trained_models/vgg16.pt"))
    else:
        models[i].load_state_dict(torch.load("./trained_models/"+model_name+".pt"))
    models[i].eval()


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
        mean=torch.zeros(len(kappa))
        for j, (x, y) in enumerate(dataloaders['test']):
            x=x.to(device)
            y=y.to(device)
            perturbed_x=adversaries[i].generate(x, y, epsilon=eps, step_size=eps/3, num_steps=10)
            for idx, k in enumerate(kappa):
                m=mean_weighted_distance(kpixel_attack(x, perturbed_x, k=k)-x)
                mean[idx]+=m
        with open(outfile_name, 'a') as outfile:
            outfile.write("\nAttack on "+model_names[i]+" mean distances: "+str(mean/len(dataloaders['test'])))
        print('model: ', model_names[i], " mean: ", mean/len(dataloaders['test']))
