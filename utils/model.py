import torch


def get_model_names():
    """
    Returns a list of all model names
    :return: list of model names
    """
    return ['resnet18', 'tv_resnet50', 'tv_resnet101', 'vgg16', 'vit_base_patch16_224', 'vit_base_patch32_224',
            'vit_small_patch16_224', 'vit_small_patch32_224']


def load_trained_models(models_dict):
    """
    Loads trained models (on imagenette) from trained models folder
    :param models_dict: dictionary of models
    """
    for model_name, model in models_dict.items():
        if 'vit' in model_name:
            model.head.load_state_dict(torch.load("./trained_models/in1k" + model_name[4:-4] + ".pt"))
        elif 'vgg' in model_name:
            model.head.fc.load_state_dict(torch.load("./trained_models/vgg16.pt"))
        else:
            model.load_state_dict(torch.load("./trained_models/" + model_name + ".pt"))
        model.eval()