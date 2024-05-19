import os
import torch
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# Helper function to download the dataset only once
def download_dataset(root='./data'):
    if not os.path.exists(os.path.join(root, 'cifar-100-python')):
        print("Downloading CIFAR-100 dataset...")
        CIFAR100(root=root, train=True, download=True)
        CIFAR100(root=root, train=False, download=True)

# Download dataset if not already done
download_dataset()

# CIFAR-100 dataset
train_transforms = tt.Compose([
    tt.RandomCrop(32, padding=4, padding_mode='reflect'),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

test_transforms = tt.Compose([
    tt.ToTensor(),
    tt.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
])

trainset = CIFAR100(root='./data', train=True, download=False, transform=train_transforms)
testset = CIFAR100(root='./data', train=False, download=False, transform=test_transforms)

trainloader = DataLoader(trainset, batch_size=400, shuffle=True, num_workers=2, pin_memory=True)
testloader = DataLoader(testset, batch_size=800, num_workers=2, pin_memory=True)

device = get_default_device()
trainloader = DeviceDataLoader(trainloader, device)
testloader = DeviceDataLoader(testloader, device)
