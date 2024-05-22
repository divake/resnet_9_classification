import os
import pickle
import numpy as np
import random
import torch
from PIL import Image
import torchvision.transforms as tt
from torch.utils.data import Dataset

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
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Normalization and transformations for CIFAR-100
transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])

# Paths
data_dir = './data/cifar-100-python/'
test_path = os.path.join(data_dir, 'test')

# Load the original test dataset
cifar100_test = CIFAR100SplitDataset(test_path, transform=transform)

# Fix the random seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Separate 100 images for calibration and the rest for testing
num_calib = 100
num_test = len(cifar100_test) - num_calib
# In the code below, we randomly select 100 indices for calibration and use the rest for testing
calib_indices = random.sample(range(len(cifar100_test)), num_calib)
test_indices = list(set(range(len(cifar100_test))) - set(calib_indices))

calibration_data = [cifar100_test.data[i] for i in calib_indices]
calibration_labels = [cifar100_test.labels[i] for i in calib_indices]
test_data = [cifar100_test.data[i] for i in test_indices]
test_labels = [cifar100_test.labels[i] for i in test_indices]

# Save the calibration and test sets in the same format as the original test set
def save_cifar100_format(data, labels, filename):
    batch = {
        'data': np.concatenate([d.reshape(1, -1) for d in data], axis=0),
        'fine_labels': labels,
        'coarse_labels': labels,  # CIFAR-100 does not use coarse labels
        'batch_label': filename,
        'filenames': [f'{i}.png' for i in range(len(labels))]
    }
    with open(filename, 'wb') as f:
        pickle.dump(batch, f)

# Save the calibration and test sets
calibration_path = os.path.join(data_dir, 'calibration')
test_set_path = os.path.join(data_dir, 'test')

save_cifar100_format(calibration_data, calibration_labels, calibration_path)
save_cifar100_format(test_data, test_labels, test_set_path)

# Print the number of images in each dataset
print(f"Number of images in the calibration set: {len(calibration_data)}")
print(f"Number of images in the test set: {len(test_data)}")
