import timm
import torch
from deeprobust.image.attack.pgd import PGD
from torchvision import transforms
from utils.data import get_dataloaders
from torchvision.utils import save_image

outfile_name="./examples/pgd/logits.txt"

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
                              test_batch_size=32,
                              data_transforms=data_transforms,
                              shuffle_test=True)

model_names=['tv_resnet101', 'vit_base_patch16_224']
#model_names=['resnet18']
epsilons=[0.005]
#epsilons=[10]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models=[timm.create_model(model_name, pretrained=True, num_classes=10).to(device) for model_name in model_names]


for i, model_name in enumerate(model_names):
    if 'vit' in model_name:
        models[i].head.load_state_dict(torch.load("./trained_models/in1k"+model_name[4:-4]+".pt"))
    elif 'vgg' in model_name:
        models[i].head.fc.load_state_dict(torch.load("./trained_models/vgg16.pt"))
    else:
        models[i].load_state_dict(torch.load("./trained_models/"+model_name+".pt"))
    models[i].eval()


adversaries=[PGD(model, 'cuda') for model in models]
for eps in epsilons:
    for n, (x, y) in enumerate(dataloaders['test']):
        if n>0:
            break
        x=x.to(device)
        y=y.to(device)
        for i, attacked_model in enumerate(models):
            out=attacked_model(x)
            perturbed_x=adversaries[i].generate(x, y, epsilon=eps, step_size=eps/3, num_steps=10)
            out_perturbed=attacked_model(perturbed_x)
            smax=torch.softmax(out_perturbed, dim=1).detach().cpu().numpy()
            out=torch.argmax(out, axis=1)
            out_perturbed=torch.argmax(out_perturbed, axis=1)
            for j in range(0, x.shape[0]):
                save_image(x[j], f'./examples/pgd/{j}_{model_names[i]}_originalREAL_{y[j].item()}_CLEAN_{out[j].item()}_ADV_{out_perturbed[j].item()}.png')
                save_image(perturbed_x[j], f'./examples/pgd/{j}_{model_names[i]}_perturbedREAL_{y[j].item()}_CLEAN_{out[j].item()}_ADV_{out_perturbed[j].item()}.png')
                save_image(perturbed_x[j]-x[j], f'./examples/pgd/{j}_{model_names[i]}_deltaREAL_{y[j].item()}_CLEAN_{out[j].item()}_ADV_{out_perturbed[j].item()}.png')
                with open(outfile_name, 'a+') as outfile:
                    outfile.write(f'\nimage{j}_{model_names[i]}_{smax[j]}')
