import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os
import pickle
from PIL import Image
import torchvision

from conformal_classification.conformal import ConformalModelLogits
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
nesting_list = [1028, 512, 256, 128, 64, 32, 16, 8]
model = ResNet9(3, 100, nesting_list=nesting_list)
model.load_state_dict(torch.load('resnet9_mrl.pth'))
model = model.cuda()
model.eval()

# Conformalize the model for each nesting resolution
conformal_models = []
for _ in nesting_list:
    cmodel = ConformalModelLogits(model, calib_loader, alpha=0.1)
    conformal_models.append(cmodel)

# Validate the coverage of the conformal model on the calibration set for each resolution
for i, cmodel in enumerate(conformal_models):
    print(f"Evaluating for nesting size: {nesting_list[i]}")
    top1, top5, coverage, size = validate(test_loader, cmodel, print_bool=True)

# Fixed indices for the specific images you want
fixed_indices = [21, 81, 62, 13, 50, 60, 75, 19]  # Replace with actual indices

# Use these fixed indices to create the subset
explore_data = Subset(cifar100_calib, fixed_indices)

# Print indices of the selected images
print(f"Selected indices: {fixed_indices}")

# Plot example predictions for each nesting resolution
num_images = 8
for i, cmodel in enumerate(conformal_models):
    mosaiclist = []
    sets = []
    labels = []
    for data in explore_data:
        img, label = data
        scores, set_pred = cmodel(img.unsqueeze(0).cuda())
        probabilities = torch.softmax(scores, dim=1).squeeze(0)  # Get probabilities
        unnormalized_img = (img * torch.tensor([0.2673, 0.2564, 0.2762]).view(-1, 1, 1)) + torch.tensor([0.5071, 0.4865, 0.4409]).view(-1, 1, 1)
        
        # Filter out classes with 0.0 probability and sort by probability
        set_pred_with_probs = [(cifar100_calib.classes[s], round(probabilities[s].item(), 2)) for s in set_pred[0] if round(probabilities[s].item(), 2) > 0.0]
        sorted_set_pred_with_probs = sorted(set_pred_with_probs, key=lambda item: item[1], reverse=True)  # Sort by probability
        
        sets.append(sorted_set_pred_with_probs)
        labels.append(cifar100_calib.classes[label])
        mosaiclist.append(unnormalized_img)
    
    grid = torchvision.utils.make_grid(mosaiclist)
    
    fig, ax = plt.subplots(figsize=(min(num_images, 9) * 5, np.floor(num_images / 9 + 1) * 5))
    ax.imshow(grid.permute(1, 2, 0), interpolation='nearest')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    
    # Calculate the size of the filtered RAPS sets
    filtered_size = sum(len(s) for s in sets) / num_images
    
    for j in range(len(mosaiclist)):
        print(f"Image {j} at nesting size {nesting_list[i]} has label '{labels[j]}', and the predictive set with probabilities is {sets[j]}.")
    
    print(f"Filtered Size@RAPS for nesting size {nesting_list[i]}: {filtered_size:.3f}")

    plt.show()
