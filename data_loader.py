import os
import numpy as np
import torchvision.transforms as tt
import torchvision
from torch.utils.data import DataLoader
from config import Config

cfg = Config()

# Paths to check if data is already downloaded
train_data_path = './data/cifar-100-python/train'
test_data_path = './data/cifar-100-python/test'

# Function to check if dataset is already downloaded
def is_data_downloaded(data_path):
    return os.path.exists(data_path)

# Download the dataset to calculate mean and std
if not is_data_downloaded(train_data_path):
    train_data = torchvision.datasets.CIFAR100('./data', train=True, download=True)
else:
    train_data = torchvision.datasets.CIFAR100('./data', train=True, download=False)

# Calculate mean and std for normalization
x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
mean = np.mean(x, axis=(0, 1)) / 255
std = np.std(x, axis=(0, 1)) / 255
mean = mean.tolist()
std = std.tolist()

# Define transformations
transform_train = tt.Compose([
    tt.RandomCrop(32, padding=4, padding_mode='reflect'),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(mean, std, inplace=True)
])
transform_test = tt.Compose([tt.ToTensor(), tt.Normalize(mean, std)])

# Load datasets
if not is_data_downloaded(train_data_path):
    trainset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
else:
    trainset = torchvision.datasets.CIFAR100('./data', train=True, download=False, transform=transform_train)

if not is_data_downloaded(test_data_path):
    testset = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)
else:
    testset = torchvision.datasets.CIFAR100('./data', train=False, download=False, transform=transform_test)

# Create data loaders
trainloader = DataLoader(trainset, cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
testloader = DataLoader(testset, cfg.batch_size * 2, pin_memory=True, num_workers=2)
