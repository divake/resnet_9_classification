import os
import pickle
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as tt
from torch.utils.data import DataLoader, Dataset
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

# Custom dataset class to handle loading of split datasets
class CIFAR100SplitDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            self.data = batch[b'data'] if b'data' in batch else batch['data']
            self.labels = batch[b'fine_labels'] if b'fine_labels' in batch else batch['fine_labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)  # Convert to uint8
        img = Image.fromarray(img)  # Convert numpy array to PIL Image
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

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

# Hardcoded paths for the datasets
data_dir = './data/cifar-100-python/'
train_path = os.path.join(data_dir, 'train')
test_path = os.path.join(data_dir, 'test')
calibration_path = os.path.join(data_dir, 'calibration')

# Load datasets using custom class for train, test, and calibration
trainset = CIFAR100SplitDataset(train_path, transform=train_transforms)
testset = CIFAR100SplitDataset(test_path, transform=test_transforms)
calibrationset = CIFAR100SplitDataset(calibration_path, transform=test_transforms)

# Create data loaders
trainloader = DataLoader(trainset, batch_size=400, shuffle=True, num_workers=2, pin_memory=True)
testloader = DataLoader(testset, batch_size=800, num_workers=2, pin_memory=True)
calib_loader = DataLoader(calibrationset, batch_size=800, shuffle=True, num_workers=2, pin_memory=True)

# Move data loaders to device
device = get_default_device()
trainloader = DeviceDataLoader(trainloader, device)
testloader = DeviceDataLoader(testloader, device)
calib_loader = DeviceDataLoader(calib_loader, device)
