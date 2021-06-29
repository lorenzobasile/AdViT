import argparse
# from deeprobust.image.defense.fgsmtraining import FGSMtraining
from utils.adversarial_train import FGSMtrain
from utils.adversarial_train import PGDtrain
# from deeprobust.image.defense.pgdtraining import PGDtraining
from torchvision import transforms
from utils.data import get_dataloaders
import torch
import timm

parser = argparse.ArgumentParser(description='Adversarial training')
parser.add_argument('--data_dir', default='data/imagenette2-320/', type=str)
parser.add_argument('--attack', default='FGSM', type=str)
parser.add_argument('--train_batch_size', default=32, type=int)
parser.add_argument('--test_batch_size', default=128, type=int)
parser.add_argument('--num_epochs', default=15, type=int)

args = parser.parse_args()

if args.attack not in ['FGSM', 'PGD']:
    raise ValueError(f"{args.attack} has no available adversarial training")

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
dataloaders = get_dataloaders(data_dir=args.data_dir,
                              train_batch_size=args.train_batch_size,
                              test_batch_size=args.test_batch_size,
                              data_transforms=data_transforms)

model_names =['vgg16', 'vit_base_patch16_224']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
models = [timm.create_model(model_name, pretrained=True, num_classes=10).to(device) for model_name in model_names]
epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]

for i, model_name in enumerate(model_names):
    for eps in epsilons:

        model = models[i].to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
        '''
        if 'vit' in model_name:
            model.head.load_state_dict(torch.load(f"trained_models/{model_name[4:-4]}.pt"))
        elif 'vgg' in model_name:
            model.head.fc.load_state_dict(torch.load("trained_models/vgg16.pt"))
        else:
            model.fc.load_state_dict(torch.load(f"trained_models/{model_name}.pt"))
        '''
        model.eval()
        
        name_model = 'vit_' + model_name[4:-4] if 'vit' in model_name else model_name

        if args.attack == 'FGSM':
            print(f"FGSM Training for {name_model}, eps={eps:.3f}")
            FGSMtrain(model, dataloaders,
                      n_epochs=10,
                      optimizer=optimizer,
                      outfile_name=f"./training_outputs/{name_model}_{args.attack}_eps{eps:.3f}.txt",
                      eps=eps,
                      clip=True)
        elif args.attack == 'PGD':
            print(f"PGD Training for {name_model}, eps={eps:.3f}")
            PGDtrain(model, dataloaders,
                     n_epochs=10,
                     optimizer=optimizer,
                     outfile_name=f"./training_outputs/{name_model}_{args.attack}_eps{eps:.3f}.txt",
                     eps=eps,
                     clip=True)

        if model_name == 'vgg16':
            torch.save(model.head.fc.state_dict(),
                       f"trained_models/{model_name}_{args.attack}_eps{eps:.3f}.pt")  # to save memory
        else:
            torch.save(model.state_dict(), f"trained_models/{model_name}_{args.attack}_eps{eps:.3f}.pt")
