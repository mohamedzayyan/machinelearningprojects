#!/usr/bin/env python3

from __future__ import print_function, division

from config import *
from GalaxiesDataset import *
from SanderDielemanNet import *

import os, time, copy, sys
import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

################################################################################
### Load the dataset
################################################################################
if MODEL == "sander_dieleman":
	transf = transforms.Compose([
	                             transforms.CenterCrop((224, 224)),
	                             transforms.Resize((45, 45)),
	                             transforms.RandomHorizontalFlip(p=0.5),
	                             transforms.RandomRotation(degrees=(0,360)),
	                             transforms.RandomVerticalFlip(p=0.5),
	                             transforms.ToTensor(),
	                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                                  std=[0.229, 0.224, 0.225])
	                           ])
else:
	transf = transforms.Compose([
	                             transforms.CenterCrop((224, 224)),
	                             transforms.RandomHorizontalFlip(p=0.5),
	                             transforms.RandomRotation(degrees=(0,360)),
	                             transforms.RandomVerticalFlip(p=0.5),
	                             transforms.ToTensor(),
	                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                                  std=[0.229, 0.224, 0.225])
	                           ])
train_ds = GalaxiesDataset(TRAIN_DIR, TRAIN_CSV, transform=transf)

size = len(train_ds)
indices = list(range(size))
split = int(np.floor(VALIDATION_SPLIT * size))
if SHUFFLE_DS:
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
val_sampler   = SubsetRandomSampler(val_indices)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=22,
                                                 sampler=train_sampler)
val_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=22,
                                                 sampler=val_sampler)
print("Total: {} Train_dl: {} Validation_dl: {}".format(size, len(train_dl),
                                                              len(val_dl)))
################################################################################



################################################################################
### Create the network and move it to the appropriate device
################################################################################
if MODEL == "sander_dieleman":
	model = SanderDielemanNet(num_classes=37)
elif MODEL == "alexnet":
	model = models.AlexNet(37)
elif MODEL == "vgg16":
	model = models.vgg16(num_classes=37)
elif MODEL == "resnet50":
	model = models.resnet50(num_classes=37)
else:
	print("Unknown MODEL requested.")
	sys.exit(1)
print("Creating", MODEL, "model")
model = nn.DataParallel(model, device_ids=range(NR_DEVICES))
device    = torch.device(DEVICE)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.4, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
################################################################################



################################################################################
### Train the network
################################################################################
def train_phase():
    model.train()
    losses = []
    epoch_start = time.time()
    for i, batch in enumerate(train_dl):
        inputs, labels = batch['image'], batch['labels'].float().view(-1,37)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()             # 1. Zero the parameter gradients
        outputs = model(inputs)           # 2. Run the model

        loss = criterion(outputs, labels) # 3. Calculate loss
        losses.append(loss.item())
        loss = torch.sqrt(loss)           #    -> RMSE loss
        loss.backward()                   # 4. Backward propagate the loss
        optimizer.step()                  # 5. Optimize the network
        
        #print("--> Batch {}/{} Loss: {}".format(i+1, len(train_dl), loss.item()))
        
    epoch_loss = np.sqrt(sum(losses) / len(losses))
    epoch_time = time.time() - epoch_start
    print("[TST] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                epoch_time // 60, 
                                                                epoch_time % 60))
    return epoch_loss

def validate_phase():
    model.eval()
    losses = []
    epoch_start = time.time()
    for i, batch in enumerate(val_dl):
        inputs, labels = batch['image'], batch['labels'].float().view(-1,37)
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)           # 2. Run the model

        loss = criterion(outputs, labels) # 3. Calculate loss
        losses.append(loss.item())
        loss = torch.sqrt(loss)           #    -> RMSE loss
        
        #print("--> Batch {}/{} Loss: {}".format(i+1, len(train_dl), loss.item()))
        
    epoch_loss = np.sqrt(sum(losses) / len(losses))
    epoch_time = time.time() - epoch_start
    print("[VAL] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                epoch_time // 60, 
                                                                epoch_time % 60))
    return epoch_loss

train_losses = []
val_losses = []
for epoch in range(NUM_EPOCHS):
    train_loss = train_phase()
    val_loss   = validate_phase()
    scheduler.step(val_loss)    
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
################################################################################
    


################################################################################
### Save the trained model
################################################################################
if SAVE_MODEL:
	torch.save(model, MODEL_FILENAME)
################################################################################



################################################################################
### Run the test and produce the output csv file
################################################################################
if MODEL == "sander_dieleman":
	transf = transforms.Compose([
	                             transforms.CenterCrop((224, 224)),
	                             transforms.Resize((45, 45)),
	                             transforms.ToTensor(),
	                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                                  std=[0.229, 0.224, 0.225])
	                            ])
else:
	transf = transforms.Compose([
	                             transforms.CenterCrop((224, 224)),
	                             transforms.ToTensor(),
	                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                                  std=[0.229, 0.224, 0.225])
	                            ])
test_ds = GalaxiesDataset(TEST_DIR, TEST_CSV, transform=transf)
test_dl = DataLoader(test_ds, batch_size=64, num_workers=22, shuffle=False)

model.eval()
for i, batch in enumerate(test_dl):
    inputs = batch['image']
    inputs = inputs.to(device)
    outputs = model(inputs)
    
    if i == 0:
        ids = batch['id'].numpy()
        labels = outputs.detach().cpu().numpy()
    else:
        ids = np.concatenate((ids, batch['id'].numpy()))
        labels = np.vstack((labels, outputs.detach().cpu().numpy()))
    
print(ids.shape, labels.shape)
combined = np.column_stack((ids, labels))
print(combined.shape)

pd_df = pd.DataFrame(combined, columns=CSV_HEADER)
pd_df["GalaxyID"] = pd_df["GalaxyID"].astype(np.uint32)
pd_df.to_csv(OUTPUT_CSV, index=None)
################################################################################
