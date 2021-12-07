import timm
import torch

from utils.data import get_dataloaders
from utils.neighbours import extract_neighbourhoods
import argparse

from utils.model import get_model_names, load_trained_models

parser = argparse.ArgumentParser(description='FGSM attack -- accuracy evaluation')
parser.add_argument('--model_folder', type=str, default='normal_training', help="model folder")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='dataset name')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
args = parser.parse_args()

# get dataloaders
dataloaders = get_dataloaders(data_dir=args.data,
                              train_batch_size=args.train_batch_size,
                              test_batch_size=args.test_batch_size)
# get device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# generate pretrained models with timm
models = {model_name: timm.create_model(model_name, pretrained=True, num_classes=10).to(device) for model_name in get_model_names()}
print("Models:", models.keys())

# load trained models on imagenette2-320
load_trained_models(models, trained_models_folder=args.model_folder)

for model_name, model in models.items():
    print(model_name)
    extract_neighbourhoods(model, dataloaders['test'])
