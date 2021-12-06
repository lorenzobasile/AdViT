import argparse

from utils.adversarial_train import ADVtrain
from utils.data import get_dataloaders
import torch
import timm

parser = argparse.ArgumentParser(description='Adversarial training')
parser.add_argument('--data_dir', default='data/imagenette2-320/', type=str)
parser.add_argument('--attack', default='fgsm', type=str)
parser.add_argument('--train_batch_size', default=32, type=int)
parser.add_argument('--test_batch_size', default=32, type=int)
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--lr', default=3e-2, type=float)

args = parser.parse_args()

if args.attack not in ['fgsm', 'pgd']:
    raise ValueError(f"{args.attack} has no available adversarial training")

dataloaders = get_dataloaders(data_dir=args.data_dir,
                              train_batch_size=args.train_batch_size,
                              test_batch_size=args.test_batch_size)

model_names = ['vit_base_patch16_224']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# generate pretrained models with timm
models = {model_name: timm.create_model(model_name, pretrained=True, num_classes=10).to(device) for model_name in
          model_names}
print("Models:", models)

epsilons = [0.0100]

for model_name, model in models.items():
    for eps in epsilons:

        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        if 'vit' in model_name:
            model.head.load_state_dict(torch.load(f"trained_models/in1k{model_name[4:-4]}.pt"))
        elif 'vgg' in model_name:
            model.head.fc.load_state_dict(torch.load("trained_models/vgg16.pt"))
        else:
            model.load_state_dict(torch.load(f"trained_models/{model_name}.pt"))

        name_model = 'vit_' + model_name[4:-4] if 'vit' in model_name else model_name

        print(f"{args.attack} Training for {name_model}, eps={eps:.4f}")
        ADVtrain(model, args.attack, dataloaders,
                 n_epochs=args.epochs,
                 optimizer=optimizer,
                 outfile_name=f"./training_outputs/{name_model}_{args.attack}_eps{eps:.4f}.txt",
                 eps=eps,
                 clip=True)

        if 'vit' in model_name:
            torch.save(model.state_dict(), f"trained_models/{model_name}_{args.attack}_eps{eps:.4f}.pt")

        elif model_name == 'vgg16':
            torch.save(model.head.fc.state_dict(),
                       f"trained_models/{model_name}_{args.attack}_eps{eps:.4f}.pt")  # to save memory
        else:
            torch.save(model.state_dict(), f"trained_models/{model_name}_{args.attack}_eps{eps:.4f}.pt")
