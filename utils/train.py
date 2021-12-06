import torch


def train(model, dataloaders, n_epochs, optimizer, scheduler=None, outfile_name=None, clip=False):
    loss=torch.nn.CrossEntropyLoss()
    device=torch.device("cuda:1" if next(model.parameters()).is_cuda else "cpu")
    if outfile_name is not None:
        with open(outfile_name, 'w') as outfile:
            outfile.write("")
    for epoch in range(n_epochs):
        if outfile_name is not None:
            with open(outfile_name, 'a') as outfile:
                outfile.write("\nEpoch: "+str(epoch+1)+'/'+str(n_epochs))
        print("Epoch: ", epoch+1, '/', n_epochs)
        model.train()
        for x, y in dataloaders['train']:
            x=x.to(device)
            y=y.to(device)
            out,_=model(x)
            l=loss(out, y)
            optimizer.zero_grad()
            l.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.eval()
        for i in ['train', 'test']:
            correct=0
            with torch.no_grad():
                for x, y in dataloaders[i]:
                    out,_=model(x.to(device))
                    correct+=(torch.argmax(out, axis=1)==y.to(device)).sum().item()
            if outfile_name is not None:
                with open(outfile_name, 'a') as outfile:
                    outfile.write("\nAccuracy on "+i+" set: "+str(correct/len(dataloaders[i].dataset)))
            print("Accuracy on "+i+" set: ", correct/len(dataloaders[i].dataset))
