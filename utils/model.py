import os

import timm
import torch
from torch import nn


def get_model_names():
    """
    Returns a list of all model names
    :return: list of model names
    """
    return ['resnet18', 'tv_resnet50', 'tv_resnet101', 'vgg16', 'vit_base_patch16_224', 'vit_base_patch32_224',
            'vit_small_patch16_224', 'vit_small_patch32_224']


def get_cnn_names():
    """
    Returns a list of all cnn names
    :return: list of cnn names
    """
    return ['resnet18', 'tv_resnet50', 'tv_resnet101',  'vgg16']


def get_vit_names():
    """
    Returns a list of all vit names
    :return: list of vit names
    """
    return ['vit_base_patch16_224', 'vit_base_patch32_224', 'vit_small_patch16_224', 'vit_small_patch32_224']


def load_trained_models(models_dict, trained_models_folder='normal_training'):
    """
    Loads trained models (on imagenette) from trained models folder
    :param models_dict: dictionary of models
    """

    # raise error if trained_models_folder is not the name of a folder inside trained_models
    if trained_models_folder not in os.listdir('trained_models'):
        raise ValueError('trained_models_folder must be the name of a folder inside trained_models')

    for model_name, model in models_dict.items():
        if trained_models_folder == "adversarial_training":
            if 'vit' in model_name:
                model.load_state_dict(torch.load(f"./trained_models/{trained_models_folder}/{model_name}_PGD_eps0.0100.pt"))
            else:
                model.load_state_dict(torch.load(f"./trained_models/{trained_models_folder}/{model_name}_PGD_eps0.0100.pt"))
        elif trained_models_folder == "normal_training":
            if 'vit' in model_name:
                model.head.load_state_dict(torch.load("./trained_models/normal_training/in1k" + model_name[4:-4] + ".pt"))
            elif 'vgg' in model_name:
                model.head.fc.load_state_dict(torch.load("./trained_models/normal_training/vgg16.pt"))
            else:
                model.load_state_dict(torch.load("./trained_models/normal_training/" + model_name + ".pt"))
        model.eval()


def create_ViT(img_size=224, patch_size=16, num_classes=10):
    """
    Creates a ViT model
    :param img_size: size of input image
    :param patch_size: size of patch
    :param num_classes: number of classes
    :return: model
    """
    weights = None

    model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=num_classes, img_size=img_size)

    if patch_size != 16:
        for p in model.named_parameters():
            if p[0] == 'patch_embed.proj.bias':
                biases = p[1]
            if p[0] == 'patch_embed.proj.weight':
                weights = p[1]
        sampling_step = 16 // patch_size
        sampled_weights = weights[:, :, ::sampling_step, ::sampling_step]

        model.patch_embed.proj = nn.Conv2d(3, 768, (patch_size, patch_size), (patch_size, patch_size))
        model.patch_embed.proj.weight = nn.Parameter(sampled_weights)
        model.patch_embed.proj.bias = nn.Parameter(biases)
        model.patch_embed.num_patches = (img_size // patch_size) ** 2
        model.pos_embed = nn.Parameter(torch.zeros(1, model.patch_embed.num_patches + 1, 768))

    return model
