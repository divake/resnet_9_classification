import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os
import pickle
from PIL import Image
import torchvision

from conformal_classification.conformal import ConformalModel
from conformal_classification.utils import validate, get_logits_targets, sort_sum

# Insert the path to the conformal_classification directory
sys.path.insert(1, os.path.join(sys.path[0], './conformal_classification/'))

# Import your ResNet9 model
from model import ResNet9

# Fix the random seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

# Normalization and transformations for CIFAR-100
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
])

batch_size = 128

# Custom dataset class to handle loading of split datasets
class CIFAR100SplitDataset(datasets.VisionDataset):
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    def __init__(self, file_path, transform=None):
        super(CIFAR100SplitDataset, self).__init__(root=file_path, transform=transform)
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            self.data = batch[b'data'] if b'data' in batch else batch['data']
            self.labels = batch[b'fine_labels'] if b'fine_labels' in batch else batch['fine_labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Load datasets
data_dir = './data/cifar-100-python/'
train_path = os.path.join(data_dir, 'train')
test_path = os.path.join(data_dir, 'test')
calibration_path = os.path.join(data_dir, 'calibration')

cifar100_train = CIFAR100SplitDataset(train_path, transform=transform)
cifar100_test = CIFAR100SplitDataset(test_path, transform=transform)
cifar100_calib = CIFAR100SplitDataset(calibration_path, transform=transform)

# Initialize data loaders
calib_loader = DataLoader(cifar100_calib, batch_size=batch_size, shuffle=True, pin_memory=True)
train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, pin_memory=True)

# Get your pre-trained model
model = ResNet9(3, 100)
model.load_state_dict(torch.load('group22_pretrained_model.h5'))
model = model.cuda()
model.eval()

# Conformalize the model
num_classes = len(cifar100_calib.classes)  # Get the number of classes
cmodel = ConformalModel(model, calib_loader, alpha=0.1, num_classes=num_classes)

# Validate the coverage of the conformal model on the test set
top1, top5, coverage, size = validate(calib_loader, cmodel, print_bool=True)

# Plot example predictions
num_images = 8
explore_data, _ = random_split(cifar100_test, [num_images, len(cifar100_test) - num_images])

mosaiclist = []
sets = []
labels = []

for data in explore_data:
    img, label = data
    scores, set_pred = cmodel(img.unsqueeze(0).cuda())
    unnormalized_img = (img * torch.tensor([0.2673, 0.2564, 0.2762]).view(-1, 1, 1)) + torch.tensor([0.5071, 0.4865, 0.4409]).view(-1, 1, 1)
    
    set_pred = [cifar100_test.classes[s] for s in set_pred[0]]
    sets.append(set_pred)
    labels.append(cifar100_test.classes[label])
    mosaiclist.append(unnormalized_img)

grid = torchvision.utils.make_grid(mosaiclist)

fig, ax = plt.subplots(figsize=(min(num_images, 9) * 5, np.floor(num_images / 9 + 1) * 5))
ax.imshow(grid.permute(1, 2, 0), interpolation='nearest')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()

for i in range(len(mosaiclist)):
    print(f"Image {i} has label '{labels[i]}', and the predictive set is {sets[i]}.")

plt.show()
