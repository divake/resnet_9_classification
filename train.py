import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import trainloader, testloader, to_device, get_default_device
from model import ResNet9
from config import Config

# Configuration setup
config = Config()
device = config.device

# Ensure the nesting list starts with the correct dimensions
nesting_list = [1028, 512, 256, 128, 64, 32, 16, 8]

# Model initialization
model = ResNet9(3, 100, nesting_list=nesting_list)
model = to_device(model, device)

# Define the Matryoshka_CE_Loss class
class Matryoshka_CE_Loss(nn.Module):
    def __init__(self, relative_importance=None, **kwargs):
        super(Matryoshka_CE_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)
        self.relative_importance = relative_importance

    def forward(self, output, target):
        losses = torch.stack([self.criterion(output_i, target) for output_i in output])
        if self.relative_importance is None:
            rel_importance = torch.ones_like(losses, device=losses.device)
        else:
            rel_importance = torch.tensor(self.relative_importance, device=losses.device)
        weighted_losses = rel_importance * losses
        return weighted_losses.sum()

# Evaluation function
@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)

# Prediction function for test labels
@torch.no_grad()
def test_label_predictions(model, device, test_loader, rep_size=None):
    model.eval()
    actuals = []
    predictions = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        if isinstance(output, list) and rep_size is not None:
            output = output[rep_size]  # Use the specified representation size for evaluation
        prediction = output.argmax(dim=1, keepdim=True)
        actuals.extend(target.view_as(prediction))
        predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

# Training function
def train_model():
    device = get_default_device()
    model = ResNet9(3, 100, nesting_list=nesting_list, efficient=False)
    model = to_device(model, device)

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = Matryoshka_CE_Loss(relative_importance=[1, 1, 1, 1, 1, 1, 1, 1])

    epochs = 106  # Adjust as needed
    for epoch in range(epochs):
        model.train()
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'resnet9_mrl.pth')

# Make sure to call the train function to start training
if __name__ == '__main__':
    train_model()
