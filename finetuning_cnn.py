import timm
import torch
from torchvision import transforms

from utils.data import get_dataloaders
from utils.train import train

#cnn_names = ['resnet18', 'tv_resnet50', 'tv_resnet101', 'vgg16']
cnn_names=['vgg16']
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
                              train_batch_size=64,
                              test_batch_size=64,
                              data_transforms=data_transforms)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



n_epochs=5
for cnn in cnn_names:
    print(cnn)
    model=timm.create_model(cnn, pretrained=True, num_classes=10)
    model=model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9)
    '''
    for p in model.named_parameters():
        p[1].requires_grad=False
        if p[0]=='fc.weight' or p[0]=='head.fc.weight' or p[0]=='fc.bias' or p[0]=='head.fc.bias':
            p[1].requires_grad=True
            if p[0] == 'fc.weight':
                final_layer = model.fc
            if p[0] == 'head.fc.weight':
	            final_layer = model.head.fc
    '''
    train(model, dataloaders, n_epochs, optimizer, outfile_name="training_outputs/"+cnn+"test.txt", clip=True)
    if cnn=='vgg16':
        torch.save(model.state_dict(), "trained_models/" + cnn + "test.pt") #to save memory
    else:
        torch.save(model.state_dict(), "trained_models/" + cnn + ".pt")
