import timm
import torch
import argparse
from utils.data import get_dataloaders
from utils.model import get_cnn_names, get_vit_names
from utils.train import train

parser = argparse.ArgumentParser(description='PyTorch ImageNette Fimetune')
parser.add_argument('--model_type', type=str, default='vit', help="network architecture")
parser.add_argument('--data', type=str, default='./data/imagenette2-320/', help='path to dataset')
parser.add_argument('--train_batch_size', type=int, default=128, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

args = parser.parse_args()

# raise value error if args.model_type is not 'cnn' nor 'vit'
if args.model_type not in ['cnn', 'vit']:
    raise ValueError('model_type must be either cnn or vit')

if args.model_type == 'cnn':
    model_names = get_cnn_names()
else:
    model_names = get_vit_names()

# get dataloaders
dataloaders = get_dataloaders(data_dir=args.data,
                              train_batch_size=args.train_batch_size,
                              test_batch_size=args.test_batch_size)

# get device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


for model_name in model_names:
    print(f'\nTraining {model_name} model...')
    model = timm.create_model(model_name, pretrained=True, num_classes=10)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    for p in model.named_parameters():
        p[1].requires_grad=False
        if p[0]=='fc.weight' or p[0]=='head.weight' or p[0]=='fc.bias' or p[0]=='head.bias':
            p[1].requires_grad=True

    train(model, dataloaders, args.epochs, optimizer, outfile_name=f"training_outputs/{model_name}.txt", clip=True)
    torch.save(model.state_dict(), "trained_models/normal_training/" + model_name + ".pt")
    '''
    if args.model_type == 'vit':
        torch.save(model.head.state_dict(), "trained_models/normal_training/" + model_name + ".pt")  # to save memory
    else:
        torch.save(model.fc.state_dict(), "trained_models/normal_training/" + model_name + ".pt")
    '''
