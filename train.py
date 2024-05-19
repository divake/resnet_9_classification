import torch
import torch.nn as nn
from data_loader import trainloader, testloader, to_device, get_default_device
from model import ResNet9, ImageClassificationBase

device = get_default_device()

# Define model
model = ResNet9(3, 100)
model = to_device(model, device)

# Evaluation function
@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)

@torch.no_grad()
def test_label_predictions(model, device, test_loader):
    model.eval()
    actuals = []
    predictions = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        prediction = output.argmax(dim=1, keepdim=True)
        actuals.extend(target.view_as(prediction))
        predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

def train_model():
    epochs = 120
    max_lr = 0.001
    grad_clip = 0.01
    weight_decay = 0.001
    opt_func = torch.optim.Adam

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def fit_one_cycle(epochs, max_lr, model, train_loader, test_loader, 
                      weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
        torch.cuda.empty_cache()
        history = []
    
        optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                    steps_per_epoch=len(train_loader))
    
        for epoch in range(epochs):
            model.train()
            train_losses = []
            lrs = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                
                if grad_clip: 
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                
                lrs.append(get_lr(optimizer))
                sched.step()
        
            result = evaluate(model, test_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            model.epoch_end(epoch, result)
            history.append(result)
        return history

    history = [evaluate(model, testloader)]
    history += fit_one_cycle(epochs, max_lr, model, trainloader, testloader, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
    # Save the model to the specified path
    torch.save(model.state_dict(), 'group22_pretrained_model.h5')
    return model

if __name__ == '__main__':
    model = train_model()
