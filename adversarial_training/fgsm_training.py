from deeprobust.image.defense.fgsmtraining import FGSMtraining
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch
import timm

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

data_dir = './data/imagenette2-320/'
datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
            ['train', 'test']}
dataloaders = {'train': DataLoader(datasets['train'], batch_size=128, shuffle=True),
               'test': DataLoader(datasets['test'], batch_size=64, shuffle=False)}

model_names = ['tv_resnet50', 'vgg16', 'vit_base_patch16_224']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models = [timm.create_model(model_name, pretrained=True, num_classes=10).to(device) for model_name in model_names]

epsilons=[0.001, 0.005, 0.01, 0.05, 0.1]

for i, model_name in enumerate(model_names):
    if 'vit' in model_name:
        models[i].head.load_state_dict(torch.load(f"../trained_models/{model_name[4:-4]}.pt"))
    elif 'vgg' in model_name:
        models[i].head.fc.load_state_dict(torch.load("./trained_models/vgg16.pt"))
    else:
        models[i].fc.load_state_dict(torch.load(f"../trained_models/{model_name}.pt"))
    models[i].eval()

for model in model_names:
    for eps in epsilons:
        name_model = model[-4:4] if 'vit' in model else model
        defense = FGSMtraining(model, device)
        defense.generate(train_loader=dataloaders['train'],
                         test_loader=dataloaders['test'],
                         save_dir=f"../adversarial_training_results",
                         save_model=True,
                         epsilon = eps,
                         save_name=f"{name_model}_adv_training_eps{eps:.2f}.pt",
                         epoch_num=5)
