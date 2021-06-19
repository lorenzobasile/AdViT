from torch.optim.lr_scheduler import LambdaLR
import math

# https://github.com/jeonsworld/ViT-pytorch
class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def train(model, dataloaders, n_epochs, optimizer, scheduler=None, outfile_name=None, clip=False)
    if outfile_name is not None:
        with open(outfile_name, 'w') as outfile:
            outfile.write("")
    for epoch in range(n_epochs):
        if outfile_name is not None:
            with open(outfile_name, 'a') as outfile:
                outfile.write("\nEpoch: "+str(epoch)+'/'+str(n_epochs))
        print("Epoch: ", epoch, '/', n_epochs)
        model.train()
        for x, y in dataloaders['train']:
            x=x.to(device)
            y=y.to(device)
            out=model(x)
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
                    out=model(x.to(device))
                    correct+=(torch.argmax(out, axis=1)==y.to(device)).sum().item()
            if outfile_name is not None:
                with open(outfile_name, 'a') as outfile:
                     outfile.write("\nAccuracy on "+i+" set: "+str(correct/len(datasets[i])))
            print("Accuracy on "+i+" set: ", correct/len(datasets[i]))
