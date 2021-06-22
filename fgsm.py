import timm
import torch
from torch.utils.data import DataLoader
from deeprobust.image.attack.fgsm import FGSM
from torchvision import transforms
import torchvision
import os

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=10)

model=model.to(device)
model.head.load_state_dict(torch.load("./trained_models/base_patch16.pt"))
model.eval()
adversary=FGSM(model, 'cuda')
correct=0
#with torch.no_grad():
for x, y in dataloaders['test']:
    x=x.to(device)
    y=y.to(device)
    x=adversary.generate(x, y, epsilon=0.1)
    out=model(x)
    correct+=(torch.argmax(out, axis=1)==y).sum().item()
print("Accuracy on test set: ", correct/len(dataloaders['test'].dataset))
