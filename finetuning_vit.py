import timm
import torch
from torchvision import transforms
from utils.data import get_dataloaders
from utils.train import train

vit_models = ['vit_base_patch16_224', 'vit_base_patch32_224', 'vit_small_patch16_224',  'vit_small_patch32_224']

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
                              test_batch_size=128,
                              data_transforms=data_transforms)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



n_epochs=5
for vit in vit_models:
    print(vit)
    model=timm.create_model(vit, pretrained=True, num_classes=10)
    model=model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)
    for p in model.named_parameters():
        p[1].requires_grad=False
        if p[0]=='head.weight' or p[0]=='head.bias':
            p[1].requires_grad=True
    train(model, dataloaders, n_epochs, optimizer, outfile_name="./training_outputs/in1k"+vit[4:-4]+".txt", clip=True)
    #train(model, dataloaders, n_epochs, optimizer, clip=True)
    torch.save(model.head.state_dict(), "./trained_models/in1k"+vit[4:-4]+".pt") #to save memory
