import torch
from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.attack.pgd import PGD

from utils.adversarial.attack import kpixel_attack


def evaluate_clean_accuracy(models_dict, dataloaders, device, outfile):
    """
    Evaluates the accuracy of the models on the test set with no adversarial attacks

    Parameters
    ----------
    models_dict : dict
        dictionary of models
    dataloaders : dict
        dictionary of dataloaders
    device : torch.device
        device to use
    outfile : str
        file to write the results to
    """
    correct = torch.zeros(len(models_dict))

    for x, y in dataloaders['test']:
        x = x.to(device)
        y = y.to(device)
        for k, (name,model) in enumerate(models_dict.items()):
            temp = torch.argmax(model(x)[0], axis=1) == y
            correct[k] += temp.sum().item()

    with open(outfile, 'w') as outfile:
        outfile.write("Clean accuracy: " + str(correct / len(dataloaders['test'].dataset)))

    print("Clean accuracy: ", correct / len(dataloaders['test'].dataset))


def evaluate_adversarial_accuracy(models_dict, dataloaders, device, outfile, epsilons=None,
                                  attack="FGSM", use_k_pixel=False, k=3000):
    """
    Evaluates the accuracy of the models on the test set with adversarial attacks

    Parameters
    ----------
    models_dict : dict
        dictionary of models
    dataloaders : dict
        dictionary of dataloaders
    device : torch.device
        device to use
    outfile : str
        file to write the results to
    epsilons : list
        list of epsilons to use for the attacks
    attack : str
        attack to use (FGSM or PGD)
    use_k_pixel : bool
        flag that tells whether to use k-pixel attack or not (only for PGD)
    k : int
        number of pixels to perturb (only for k-pixel attack)
    """
    correct_k=None

    if epsilons is None:
        epsilons = [0.0005, 0.001, 0.005, 0.01]

    # define attacks to apply on every model
    if attack == "FGSM":
        adversaries = {model_name: FGSM(model, 'cuda') for model_name, model in models_dict.items()}
    elif attack == "PGD":
        adversaries = {model_name: PGD(model, 'cuda') for model_name, model in models_dict.items()}
    else:
        raise ValueError("Attack not supported")

    # Loop on every epsilon
    for eps in epsilons:

        with open(outfile, 'a') as outfile:
            outfile.write("\nEpsilon: " + str(eps))

        # Loop on every model
        for model_name, attacked_model in models_dict.items():

            correct = torch.zeros(len(models_dict))
            if use_k_pixel:
                correct_k = torch.zeros(len(models_dict))

            for x, y in dataloaders['test']:
                x = x.to(device)
                y = y.to(device)

                # Adversarial example generation with model's attack
                if attack == "FGSM":
                    perturbed_x = adversaries[model_name].generate(x, y, epsilon=eps)
                else:
                    perturbed_x = adversaries[model_name].generate(x, y, epsilon=eps/3, steps=10)

                # Loop on every model and evaluate accuracy wrt adversarial example
                for k, model in enumerate(models_dict):
                    temp = torch.argmax(model(perturbed_x)[0], axis=1) == y
                    correct[k] += temp.sum().item()

                # K Pixel Attack
                if use_k_pixel:
                    perturbed_x = kpixel_attack(x, perturbed_x, k=k)

                    for k, model in enumerate(models_dict):
                        temp = torch.argmax(model(perturbed_x)[0], axis=1) == y
                        correct_k[k] += temp.sum().item()

            # Logging results on outfile
            with open(outfile, 'a') as outfile:
                outfile.write(f"\n{attack} attack on " + model_name + ": " + str(correct / len(dataloaders['test'].dataset)))
                outfile.write("\nk-Attack on "+model_name+": "+str(correct_k/len(dataloaders['test'].dataset)))

            print(f"{attack} attack on {model_name}: {correct / len(dataloaders['test'].dataset)}")
