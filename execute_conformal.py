import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
from tqdm import tqdm
from conformal_classification.conformal import ConformalModel
from conformal_classification.utils import validate, get_logits_targets, sort_sum
# Insert the path to the conformal_classification directory
sys.path.insert(1, os.path.join(sys.path[0], './conformal_classification/'))

from conformal_classification.conformal import ConformalModel
from conformal_classification.utils import validate 
# from conformal import ConformalModel
# from utils import validate

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

# Load CIFAR-100 dataset
cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Split the training set into a conformal calibration set and a training set
num_calib = 5000
cifar100_calib, cifar100_train = random_split(cifar100_train, [num_calib, len(cifar100_train) - num_calib])

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
cmodel = ConformalModel(model, calib_loader, alpha=0.1, lamda_criterion='size')

# Validate the coverage of the conformal model on the test set
top1, top5, coverage, size = validate(test_loader, cmodel, print_bool=True)

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
