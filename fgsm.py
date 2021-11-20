import timm
import torch

from utils.adversarial import evaluate_clean_accuracy, evaluate_adversarial_accuracy
from utils.data import get_dataloaders
import argparse

from utils.model import get_model_names, load_trained_models

parser = argparse.ArgumentParser(description='FGSM attack -- accuracy evaluation')
parser.add_argument('--outfile', type=str, default="./attack_results/fgsm.txt", help='output file')
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='dataset name')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')

args = parser.parse_args()

dataloaders = get_dataloaders(data_dir=args.data,
                              train_batch_size=args.train_batch_size,
                              test_batch_size=args.test_batch_size)

epsilons = [0.0005, 0.001, 0.005, 0.01]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
models = {model_name: timm.create_model(model_name, pretrained=True, num_classes=10).to(device) for model_name in get_model_names()}
print("Models:", models)

load_trained_models(models)

evaluate_clean_accuracy(models_dict=models, dataloaders=dataloaders, device=device, outfile=args.outfile)
evaluate_adversarial_accuracy(models_dict=models, dataloaders=dataloaders, device=device, outfile=args.outfile, epsilons=epsilons, attack="FGSM")
