import argparse
from deeprobust.image.defense.fgsmtraining import FGSMtraining
from deeprobust.image.defense.pgdtraining import PGDtraining
from torchvision import transforms
from datautils.data import get_dataloaders
import torch
import timm

parser = argparse.ArgumentParser(description='Adversarial training')
parser.add_argument('--data_dir', default='data/imagenette2-320/', type=str)
parser.add_argument('--attack', default='FGSM', type=str)
parser.add_argument('--train_batch_size', default=64, type=int)
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


model_names = ['tv_resnet50', 'vgg16']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epsilons=[0.001, 0.005, 0.01, 0.05, 0.1]

for i, model_name in enumerate(model_names):
    for eps in epsilons:

        model = timm.create_model(model_name, pretrained=True, num_classes=10).to(device)

        if 'vit' in model_name:
            model.head.load_state_dict(torch.load(f"trained_models/{model_name[4:-4]}.pt"))
        elif 'vgg' in model_name:
            model.head.fc.load_state_dict(torch.load("trained_models/vgg16.pt"))
        else:
            model.fc.load_state_dict(torch.load(f"trained_models/{model_name}.pt"))
        model.eval()

        name_model = 'vit_'+model_name[4:-4] if 'vit' in model_name else model_name

        if args.attack == 'FGSM':
            print(f"FGSM Training for {name_model}, eps={eps:.3f}")
            defense = FGSMtraining(model, device)
            defense.generate(train_loader=dataloaders['train'],
                             test_loader=dataloaders['test'],
                             save_dir=f"adversarial_training_results",
                             save_model=True,
                             epsilon = eps,
                             save_name=f"{name_model}_adv_training_fgsm_eps{eps:.3f}.pt",
                             epoch_num=args.num_epochs)
        elif args.attack == 'PGD':
            print(f"PGD Training for {name_model}, eps={eps:.3f}")

            defense = PGDtraining(model, device)
            defense.generate(train_loader=dataloaders['train'],
                             test_loader=dataloaders['test'],
                             save_dir=f"adversarial_training_results",
                             save_model=True,
                             epsilon=eps,
                             num_steps=10,
                             perturb_step_size=eps / 3,
                             save_name=f"{name_model}_adv_training_pgd_eps{eps:.3f}.pt",
                             epoch_num=15)

