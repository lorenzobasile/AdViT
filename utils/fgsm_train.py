import torch
from deeprobust.image.attack.fgsm import FGSM


def FGSMtrain(model, dataloaders, n_epochs, optimizer, eps, scheduler=None, outfile_name=None, clip=False):

    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if outfile_name is not None:
        with open(outfile_name, 'w') as outfile:
            outfile.write("")
    adversary = FGSM(model, 'cuda')


    for epoch in range(n_epochs):

        if outfile_name is not None:
            with open(outfile_name, 'a') as outfile:
                outfile.write("\nEpoch: "+str(epoch)+'/'+str(n_epochs))
        print("Epoch: ", epoch, '/', n_epochs)

        model.train()
        for x, y in dataloaders['train']:
            x=x.to(device)
            y=y.to(device)
            x_adv = adversary.generate(x, y, epsilon=eps)
            out=model(x_adv)
            l=loss(out, y)
            optimizer.zero_grad()
            l.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in dataloaders['train']:
                out = model(x.to(device))
                correct += (torch.argmax(out, axis=1) == y.to(device)).sum().item()
        print(f"\n\nClean Accuracy on training set: {correct / len(dataloaders['train'].dataset) * 100:.5f} %")

        correct_adv = 0
        for x, y in dataloaders['train']:
            out_adv = model(adversary.generate(x.to(device), y.to(device), epsilon=eps))
            correct_adv += (torch.argmax(out_adv, axis=1) == y.to(device)).sum().item()
        print(f"Adversarial Accuracy on training set: {correct_adv / len(dataloaders['train'].dataset) * 100:.5f} %")

        correct = 0
        with torch.no_grad():
            for x, y in dataloaders['test']:
                out = model(x.to(device))
                correct += (torch.argmax(out, axis=1) == y.to(device)).sum().item()
        print(f"\nClean Accuracy on test set: {correct / len(dataloaders['test'].dataset) * 100:.5f} %")

        correct_adv = 0
        for x, y in dataloaders['test']:
            out_adv = model(adversary.generate(x.to(device), y.to(device), epsilon=eps))
            correct_adv += (torch.argmax(out_adv, axis=1) == y.to(device)).sum().item()
        print(f"Adversarial Accuracy on test set: {correct_adv / len(dataloaders['test'].dataset) * 100:.5f} %")