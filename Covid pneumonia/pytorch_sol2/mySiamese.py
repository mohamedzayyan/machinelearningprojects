import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
from PIL import Image

from config import *
from GalaxiesDataset import *
from rsa_loader import *
from efigi_loader import *
from pytorchtools import EarlyStopping

import pickle
from DatasetFromSubset import *
from samplers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

class mySiamese:
    def __init__(self,train_dl, val_dl, unseen_dl, model, optimizer, scheduler, mining='hard', margin=0.2, outputSize=37,
                 modelType='triplet', device='cuda', BATCH_SIZE=64):
        
        self.margin = margin
        self.modelType = modelType
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.unseen_dl = unseen_dl
        self.modeltype = modelType
        self.BATCH_SIZE = BATCH_SIZE
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.outputSize = outputSize
        self.mining = mining

	
    def genTriplets(self, batch):
        batch_size = batch['image'].shape[0]
        im_width = batch['image'].shape[2]
        in_channels = batch['image'].shape[1]
        im_height = batch['image'].shape[3]
        
        labels = np.zeros(batch_size)
        for i in range(batch_size):
            labels[i] = batch['labels'][i].argmax()
        
        batch_img_anchor = np.zeros((batch_size, in_channels*im_width*im_height))   # image set anchors
        batch_img_positive = np.zeros((batch_size, in_channels*im_width*im_height))   # image set positive
        batch_img_negative = np.zeros((batch_size, in_channels*im_width*im_height))   # image set negatives
        
        batch_label_anchor = np.zeros((batch_size, ))    # labels for anchors
        batch_label_positive = np.zeros((batch_size,))     # labels for positives
        batch_label_negative = np.zeros((batch_size,))     # labels for negatives
        
        for i in range(batch_size):
            l = labels[i]
            #Add anchor
            batch_img_anchor[i] = torch.reshape(batch['image'][i], (in_channels*im_width*im_height, ))
            batch_label_anchor[i] = l
            
            # find and add a genuine sample
            ind_positive = np.squeeze(np.argwhere(labels == l))
            randSamp = random.sample(list(ind_positive), 1)
            batch_img_positive[i] = torch.reshape(batch['image'][randSamp], (in_channels*im_width*im_height, ))
            batch_label_positive[i] = l
            
            
            # find and add a negative sample
            ind_negative = np.squeeze(np.argwhere(labels != l))
            randSamp = random.sample(list(ind_negative), 1)
            batch_img_negative[i] = torch.reshape(batch['image'][randSamp], (in_channels*im_width*im_height, ))
            batch_label_negative[i] = labels[randSamp]
            
        batch_img_anchor = batch_img_anchor.reshape((-1, in_channels, im_width, im_height))
        batch_img_positive = batch_img_positive.reshape((-1, in_channels, im_width, im_height))
        batch_img_negative = batch_img_negative.reshape((-1, in_channels, im_width, im_height))
        
        batch_label_anchor = torch.from_numpy(batch_label_anchor).long()  # convert the numpy array into torch tensor
        batch_label_positive = torch.from_numpy(batch_label_positive).long()  # convert the numpy array into torch tensor
        batch_label_negative = torch.from_numpy(batch_label_negative).long()  # convert the numpy array into torch tensor
        
        batch_img_anchor = torch.from_numpy(batch_img_anchor).float()     # convert the numpy array into torch tensor
        batch_img_positive = torch.from_numpy(batch_img_positive).float()  # convert the numpy array into torch tensor
        batch_img_negative = torch.from_numpy(batch_img_negative).float()  # convert the numpy array into torch tensor
        
        return batch_img_anchor, batch_img_positive, batch_img_negative, batch_label_anchor, batch_label_positive, batch_label_negative
    
    def hardTriplets(self, batch_img_anchor, batch_img_positive, batch_img_negative, batch_label_anchor, 
                 batch_label_positive, batch_label_negative):
        batch_size = batch_img_anchor.shape[0]
        im_width = batch_img_anchor.shape[2]
        in_channels = batch_img_anchor.shape[1]
        im_height = batch_img_anchor.shape[3]
        
        hard_anchors_img = torch.empty((0, in_channels,im_width,im_height))
        hard_positives_img = torch.empty((0, in_channels,im_width,im_height))
        hard_negatives_img = torch.empty((0, in_channels,im_width,im_height))
        
        hard_anchors_label = torch.empty((0, 1))
        hard_positives_label = torch.empty((0, 1))
        hard_negatives_label = torch.empty((0, 1))
        
        d = nn.PairwiseDistance(p=2)
        for i in range(batch_size):
            features_a = self.model(batch_img_anchor[i].reshape((-1, in_channels, im_width, im_height)).cuda())
            features_p = self.model(batch_img_positive[i].reshape((-1, in_channels, im_width, im_height)).cuda())
            features_n = self.model(batch_img_negative[i].reshape((-1, in_channels, im_width, im_height)).cuda())
            
            d_p = d(features_a, features_p)
            d_n = d(features_a, features_n)
            
            if d_p >= d_n:
                hard_anchors_img = torch.cat((hard_anchors_img, batch_img_anchor[i].reshape((-1, in_channels, im_width, im_height))), 0)
                hard_positives_img = torch.cat((hard_positives_img, batch_img_positive[i].reshape((-1, in_channels, im_width, im_height))), 0)
                hard_negatives_img = torch.cat((hard_negatives_img, batch_img_negative[i].reshape((-1, in_channels, im_width, im_height))), 0)
                
                hard_anchors_label = torch.cat((hard_anchors_label, batch_label_anchor[i].reshape((-1, 1)).float()), 0)
                hard_positives_label = torch.cat((hard_positives_label, batch_label_positive[i].reshape((-1, 1)).float()), 0)
                hard_negatives_label = torch.cat((hard_negatives_label, batch_label_negative[i].reshape((-1, 1)).float()), 0)
            
            del features_a
            del features_p
            del features_n
            torch.cuda.empty_cache()
    
        return hard_anchors_img, hard_positives_img, hard_negatives_img, hard_anchors_label, hard_positives_label, hard_negatives_label
    
    def genPair(self, batch):
        batch_size = batch['image'].shape[0]
        im_width = batch['image'].shape[2]
        in_channels = batch['image'].shape[1]
        im_height = batch['image'].shape[3]
        
        labels = np.zeros(batch_size)
        for i in range(batch_size):
            labels[i] = batch['labels'][i].argmax()
        
        batch_img_1 = np.zeros((2 * batch_size, in_channels*im_width*im_height))   # image set 1
        batch_img_2 = np.zeros((2 * batch_size, in_channels*im_width*im_height))   # image set 2
        batch_label_1 = np.zeros((2 * batch_size, ))    # labels for image set 1
        batch_label_2 = np.zeros((2 * batch_size,))     # labels for image set 2
        batch_label_c = np.zeros((2 * batch_size,))     # contrastive label: 0 if genuine pair, 1 if impostor pair
        
        for i in range(batch_size):
            l = labels[i]
            # find and add a genuine sample
            ind_g = np.squeeze(np.argwhere(labels == l))
            batch_img_1[2*i] = torch.reshape(batch['image'][i], (in_channels*im_width*im_height, ))
            #print('label - {} #similars - {}'.format(labels[i], len(ind_g)))
            randSamp = random.sample(list(ind_g), 1)
            batch_img_2[2*i] = torch.reshape(batch['image'][randSamp], (in_channels*im_width*im_height, ))
            batch_label_1[2*i] = l
            batch_label_2[2*i] = l
            batch_label_c[2*i] = 0
            
            # find and add an impostor sample
            ind_d = np.squeeze(np.argwhere(labels != l))
            randSamp = random.sample(list(ind_d), 1)
            batch_img_1[2*i+1] = torch.reshape(batch['image'][i], (in_channels*im_width*im_height, ))
            batch_img_2[2*i+1] = torch.reshape(batch['image'][randSamp], (in_channels*im_width*im_height, ))
            batch_label_1[2*i+1] = l
            batch_label_2[2*i+1] = labels[randSamp]
            batch_label_c[2*i+1] = 1
            
        batch_img_1 = batch_img_1.reshape((-1, in_channels, im_width, im_height))
        batch_img_2 = batch_img_2.reshape((-1, in_channels, im_width, im_height))
        
        batch_label_1 = torch.from_numpy(batch_label_1).long()  # convert the numpy array into torch tensor
        #batch_label_1 = Variable(batch_label_1).cuda()          # create a torch variable and transfer it into GPU
    
        batch_label_2 = torch.from_numpy(batch_label_2).long()  # convert the numpy array into torch tensor
        #batch_label_2 = Variable(batch_label_2).cuda()          # create a torch variable and transfer it into GPU
    
        batch_label_c = batch_label_c.reshape((-1, 1))
        batch_label_c = torch.from_numpy(batch_label_c).float()  # convert the numpy array into torch tensor
        #batch_label_c = Variable(batch_label_c).cuda()           # create a torch variable and transfer it into GPU
    
        batch_img_1 = torch.from_numpy(batch_img_1).float()     # convert the numpy array into torch tensor
        #batch_img_1 = Variable(batch_img_1).cuda()              # create a torch variable and transfer it into GPU
    
        batch_img_2 = torch.from_numpy(batch_img_2).float()  # convert the numpy array into torch tensor
        #batch_img_2 = Variable(batch_img_2).cuda()           # create a torch variable and transfer it into GPU
        return batch_img_1, batch_img_2, batch_label_1, batch_label_2, batch_label_c
    
    def hardPairs(self, img_1, img_2, label_1, label_2, 
                 label_c):
        batch_size = img_1.shape[0]
        im_width = img_1.shape[2]
        in_channels = img_1.shape[1]
        im_height = img_1.shape[3]
        
        hard_img_1 = torch.empty((0, in_channels,im_width,im_height))
        hard_img_2 = torch.empty((0, in_channels,im_width,im_height))
        
        hard_label_1 = torch.empty((0, 1))
        hard_label_2 = torch.empty((0, 1))
        hard_label_c = torch.empty((0, 1))
        
        d = nn.PairwiseDistance(p=2)
        for i in range(0, batch_size, 2):
            features_a = self.model(img_1[i].reshape((-1, in_channels, im_width, im_height)).cuda())
            features_p = self.model(img_2[i].reshape((-1, in_channels, im_width, im_height)).cuda())
            features_n = self.model(img_2[i+1].reshape((-1, in_channels, im_width, im_height)).cuda())
            
            d_p = d(features_a, features_p)
            d_n = d(features_a, features_n)
            
            if d_p >= d_n:
                hard_img_1 = torch.cat((hard_img_1, img_1[i].reshape((-1, in_channels, im_width, im_height))), 0)
                hard_img_1 = torch.cat((hard_img_1, img_1[i+1].reshape((-1, in_channels, im_width, im_height))), 0)
    
                hard_img_2 = torch.cat((hard_img_2, img_2[i].reshape((-1, in_channels, im_width, im_height))), 0)
                hard_img_2 = torch.cat((hard_img_2, img_2[i+1].reshape((-1, in_channels, im_width, im_height))), 0)
                
                hard_label_1 = torch.cat((hard_label_1, label_1[i].reshape((-1, 1)).float()), 0)
                hard_label_1 = torch.cat((hard_label_1, label_1[i+1].reshape((-1, 1)).float()), 0)
                
                hard_label_2 = torch.cat((hard_label_2, label_2[i].reshape((-1, 1)).float()), 0)
                hard_label_2 = torch.cat((hard_label_2, label_2[i+1].reshape((-1, 1)).float()), 0)
                
                hard_label_c = torch.cat((hard_label_c, label_c[i].reshape((-1, 1)).float()), 0)
                hard_label_c = torch.cat((hard_label_c, label_c[i+1].reshape((-1, 1)).float()), 0)
            
            del features_a
            del features_p
            del features_n
            torch.cuda.empty_cache()
    
        return hard_img_1, hard_img_2, hard_label_1, hard_label_2, hard_label_c
    
    def triplet_loss(self, a, p, n) : 
        d = nn.PairwiseDistance(p=2)
        distance = d(a, p) - d(a, n) + self.margin 
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss
    def contrastive_loss(self, features_1, features_2, label_c):
        d = nn.PairwiseDistance(p=2)
        distance = d(features_1, features_2)
        loss_contrastive = torch.mean(0.5*(label_c) * distance +
                                         0.5* (1-label_c) *(torch.clamp(self.margin - distance, min=0.0)))
        return loss_contrastive
    
    def  train_network(self, first_images, second_images, first_labels, second_labels, c_labels, sample_size, tr='train'):
        loss_log = []    
        if tr == 'train':
            self.model.train()
        elif tr == 'val':
            self.model.eval()
        for i in range(0, sample_size, self.BATCH_SIZE):
            img_1 = first_images[i: i + self.BATCH_SIZE].cuda()
            img_2 = second_images[i: i + self.BATCH_SIZE].cuda()
            label_1 = first_labels[i: i + self.BATCH_SIZE].cuda()
            label_2 = second_labels[i: i + self.BATCH_SIZE].cuda()
            label_c = c_labels[i: i + self.BATCH_SIZE].cuda()
            
            # Reset gradients
            self.optimizer.zero_grad()
    
            # Forward pass
            features_1 = self.model(img_1)
            features_2 = self.model(img_2)
    
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            #euclidean_distance = cos(features_1, features_2)
            #euclidean_distance = F.pairwise_distance(features_1, features_2)
            #loss_contrastive = 0.5 * torch.mean((1 - label_c) * torch.pow(euclidean_distance, 2) +
            #                              label_c * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
            loss_contrastive = self.contrastive_loss(features_1, features_2, label_c)
            #torch.mean(0.5*(label_c) * euclidean_distance +
            #                            0.5* (1-label_c) *(torch.clamp(self.margin - euclidean_distance, min=0.0)))
    
            loss_log.append(float(loss_contrastive.data))
            #print('\nepoch: {} - batch: {}'.format(ep, counter))
            #print('Contrastive loss: ', float(loss_contrastive.data))
            
            # Backward pass and updates
            loss_contrastive.backward()                     # calculate the gradients (backpropagation)
            self.optimizer.step()                    # update the weights
            del img_1
            del img_2
            del label_1
            del label_2
            del label_c
            torch.cuda.empty_cache()
        epoch_loss = sum(loss_log)/len(loss_log)
        return epoch_loss
    
    def  train_network_triplets(self,anchors, positives, negatives, anchor_labels,
                            positive_labels, negative_labels, sample_size, tr='train'):
        loss_log = []    
        if tr == 'train':
            self.model.train()
        elif tr == 'val':
            self.model.eval()
        for i in range(0, sample_size, self.BATCH_SIZE):
            a = anchors[i: i + self.BATCH_SIZE].cuda()
            p = positives[i: i + self.BATCH_SIZE].cuda()
            n = negatives[i: i + self.BATCH_SIZE].cuda()
            
            label_a = anchor_labels[i: i + self.BATCH_SIZE].cuda()
            label_p = positive_labels[i: i + self.BATCH_SIZE].cuda()
            label_n = negative_labels[i: i + self.BATCH_SIZE].cuda()
            
            # Reset gradients
            self.optimizer.zero_grad()
    
            # Forward pass
            features_a = self.model(a)
            features_p = self.model(p)
            features_n = self.model(n)
    
            t_loss = self.triplet_loss(features_a, features_p, features_n)
    
            loss_log.append(float(t_loss.data))
            #print('\nepoch: {} - batch: {}'.format(ep, counter))
            #print('Contrastive loss: ', float(loss_contrastive.data))
            
            # Backward pass and updates
            t_loss.backward()                     # calculate the gradients (backpropagation)
            self.optimizer.step()                    # update the weights
            del a
            del p
            del n
            del label_a
            del label_p
            del label_n
            del features_a
            del features_p
            del features_n
            torch.cuda.empty_cache()
        epoch_loss = sum(loss_log)/len(loss_log)
        return epoch_loss
    
    def train(self, epochs=5):
        if self.outputSize != 37:
            self.model.fc[3] = nn.Linear(512, self.outputSize)
        self.model = self.model.to(self.device)
        if self.modelType == 'triplet':
            for i, batch in enumerate(self.train_dl):  # batches loop
                self.train_img_anchor, self.train_img_positve, self.train_img_negative, self.train_label_anchor, self.train_label_positive, self.train_label_negative = self.genTriplets(batch)
                
            for i, batch in enumerate(self.val_dl):  # batches loop
                self.val_img_anchor, self.val_img_positive, self.val_img_negative, self.val_label_anchor, self.val_label_positive, self.val_label_negative = self.genTriplets(batch)
            
            if self.mining == 'random':
                self.val_sample_size = self.val_img_anchor.shape[0]
                self.train_sample_size = self.train_img_anchor.shape[0]
            if self.mining == 'hard':
                self.train_hard_anchors_img, self.train_hard_positives_img, self.train_hard_negatives_img, self.train_hard_anchors_labels, self.train_hard_positives_label, self.train_hard_negatives_label = self.hardTriplets(self.train_img_anchor, self.train_img_positve, self.train_img_negative, self.train_label_anchor, self.train_label_positive, self.train_label_negative)
                self.val_hard_anchors_img, self.val_hard_positives_img, self.val_hard_negatives_img, self.val_hard_anchors_labels, self.val_hard_positives_label, self.val_hard_negatives_label = self.hardTriplets(self.val_img_anchor, self.val_img_positive, self.val_img_negative, self.val_label_anchor, self.val_label_positive, self.val_label_negative)
                
                self.train_sample_size = self.train_hard_anchors_img.shape[0]
                self.val_sample_size = self.val_hard_anchors_img.shape[0]
            train_losses=[]
            val_losses=[]
            counter = 0
            pth = './siameseEfigiTriplet/efigicheckpoint'
            early_stopping = EarlyStopping(patience=10, verbose=True, path=pth)
            
            print("Training starting")
            for ep in range(epochs):  # epochs loop
                counter += 1
                if self.mining == 'random':
                    train_loss = self.train_network_triplets(self.train_img_anchor, self.train_img_positve, self.train_img_negative,
                                                   self.train_label_anchor, self.train_label_positive,
                                                    self.train_label_negative, self.train_sample_size, tr='train' )
                    val_loss = self.train_network_triplets(self.val_img_anchor, self.val_img_positive, self.val_img_negative, self.val_label_anchor,
                                                 self.val_label_positive, self.val_label_negative, self.val_sample_size, tr='val')
                    
                if self.mining == 'hard':
                    train_loss = self.train_network_triplets(self.train_hard_anchors_img, self.train_hard_positives_img, self.train_hard_negatives_img,
                                                             self.train_hard_anchors_labels, self.train_hard_positives_label, 
                                                             self.train_hard_negatives_label, self.train_sample_size, tr='train' )
                    val_loss = self.train_network_triplets(self.val_hard_anchors_img, self.val_hard_positives_img, self.val_hard_negatives_img, 
                                                           self.val_hard_anchors_labels, self.val_hard_positives_label, self.val_hard_negatives_label , 
                                                           self.val_sample_size, tr='val')
                    
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                self.scheduler.step(val_loss) 
                    
                print('\nepoch: {}'.format(ep))
                print('Train loss: {}, Val loss: {}'.format(train_loss, val_loss) )
                
                early_stopping(val_loss, self.model, ep)
                if early_stopping.early_stop:
                    ep = ep - 10
                    self.model.load_state_dict(torch.load('./siameseEfigiTriplet/efigicheckpoint{}.pt'.format(ep)))
                    print("Early stopping")
                    break
                pickle.dump(train_losses,open('./losses/triplet_train_losses_siamese_efigi','wb'))
                pickle.dump(val_losses,open('./losses/triplet_val_losses_siamese_efigi','wb'))
            
            print("Training complete")  
            torch.save(self.model, './models/efigiSiameseTriplet')
            print('Model saved: ./models/efigiSiameseTriplet')
            
        elif self.modelType == 'pair':
            for i, batch in enumerate(self.train_dl):  # batches loop
                self.train_img_1, self.train_img_2, self.train_label_1, self.train_label_2, self.train_label_c = self.genPair(batch)
            
            for i, batch in enumerate(self.val_dl):  # batches loop
                self.val_img_1, self.val_img_2, self.val_label_1, self.val_label_2, self.val_label_c = self.genPair(batch)
                
            if self.mining == 'random':
                self.val_sample_size = self.val_img_1.shape[0]
                self.train_sample_size = self.train_img_1.shape[0]
            if self.mining == 'hard':
                self.train_hard_img_1, self.train_hard_img_2, self.train_hard_label_1, self.train_hard_label_2, self.train_hard_label_c = self.hardPairs(self.train_img_1, self.train_img_2, self.train_label_1, self.train_label_2, self.train_label_c)
                self.val_hard_img_1, self.val_hard_img_2, self.val_hard_label_1, self.val_hard_label_2, self.val_hard_label_c = self.hardPairs(self.val_img_1, self.val_img_2, self.val_label_1, self.val_label_2, self.val_label_c)
                
                self.val_sample_size = self.val_hard_img_1.shape[0]
                self.train_sample_size = self.train_hard_img_1.shape[0]
            train_losses=[]
            val_losses=[]
            counter = 0
            pth = './siameseEfigiPair/efigicheckpoint'
            early_stopping = EarlyStopping(patience=10, verbose=True, path=pth)
            print("Training starting")
            for ep in range(epochs):  # epochs loop
                counter += 1
                #train_loss = train_network(train_img_1, train_img_2, train_label_1, 
                #                               train_label_2, train_label_c, train_sample_size)
                if self.mining == 'random':
                    train_loss = self.train_network(self.train_img_1, self.train_img_2, self.train_label_1,
                                                self.train_label_2, self.train_label_c, self.train_sample_size, tr='train')
                                    
                    val_loss = self.train_network(self.val_img_1, self.val_img_2, self.val_label_1, 
                                             self.val_label_2, self.val_label_c, self.val_sample_size, tr='val')
                
                if self.mining == 'hard':
                    train_loss = self.train_network(self.train_hard_img_1, self.train_hard_img_2, self.train_hard_label_1, self.train_hard_label_2, 
                                                    self.train_hard_label_c, self.train_sample_size, tr='train')
                                    
                    val_loss = self.train_network(self.val_hard_img_1, self.val_hard_img_2, self.val_hard_label_1, self.val_hard_label_2, 
                                                  self.val_hard_label_c, self.val_sample_size,  tr='val')
                    
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                self.scheduler.step(val_loss) 
                    
                print('\nepoch: {}'.format(ep))
                print('Train loss: {}, Val loss: {}'.format(train_loss, val_loss) )
                
                early_stopping(val_loss, self.model, ep)
                if early_stopping.early_stop:
                    ep = ep - 10
                    self.model.load_state_dict(torch.load('./siameseEfigiPair/efigicheckpoint{}.pt'.format(ep)))
                    print("Early stopping")
                    break
                pickle.dump(train_losses,open('./losses/pair_train_losses_siamese_efigi','wb'))
                pickle.dump(val_losses,open('./losses/pair_val_losses_siamese_efigi','wb'))
            print("Training complete")
            torch.save(self.model, './models/efigiSiamesePair')
            print('Model saved: ./models/efigiSiamesePair')
            
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()
        
    def show_cf(self, preds, acts, cf_type):
        cf = confusion_matrix(preds, acts)
        C = cf / cf.astype(np.float).sum(axis=1, keepdims=True)
        if cf_type == 'numbers':
            print("Confusion matrix:\n%s" % cf)
        if cf_type == 'percentage':
            print("Confusion matrix:\n%s" % C) 
                
    def evaluate(self, f1_='weighted', cf_type='numbers'):
        if self.modelType == 'triplet':
            print('Evaluation beginning')
            n = len(self.unseen_dl.dataset.subset)
            m = self.train_sample_size
            self.distMatrix = torch.empty((n, m))
            self.predLabels = torch.empty((n, 1))
            self.actLabels = torch.empty((0, 1))
            self.trainPreds = torch.empty((0, self.outputSize))
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            d = nn.PairwiseDistance(p=2)
            #euclidean_distance = cos(features_1, features_2)
            self.model.eval()
            for i, batch in enumerate(self.unseen_dl):
                with torch.no_grad():
                    testPreds = self.model(batch['image'].cuda())
                    #actLabels = torch.argmax(batch['labels'], 1).float().reshape(-1,1))
                    self.actLabels = torch.cat((self.actLabels, torch.argmax(batch['labels'], 1).float().reshape(-1,1)), 0)
            for j in range(0, m, self.BATCH_SIZE):
                with torch.no_grad():
                    if self.mining == 'random':
                        preds = self.model(self.train_img_anchor[j: j + self.BATCH_SIZE].cuda())
                    if self.mining == 'hard':
                        preds = self.model(self.train_hard_anchors_img[j: j + self.BATCH_SIZE].cuda())
                    self.trainPreds = torch.cat((self.trainPreds, preds.cpu()), 0)
            for row in range(0, n):
                for col in range(0, m):
                   self.distMatrix[row, col] = d(testPreds[row].reshape(1,-1).cuda(),
                                                               self.trainPreds[col].reshape(1,-1).cuda()).cpu()
                   torch.cuda.empty_cache()
            for r in range(0, n):
                ind = torch.argmin(self.distMatrix[r])
                if self.mining == 'random':
                    self.predLabels[r, 0] = self.train_label_anchor[ind]
                if self.mining == 'hard':
                    self.predLabels[r, 0] = self.train_hard_anchors_labels[ind]
                
            preds = self.predLabels.numpy()
            acts = self.actLabels.numpy()
            df = pd.DataFrame(preds, columns=['predict'])
            df['actuals'] = acts
            print('Unseen test set accuracy: {}'.format(df[df['predict'] == df['actuals']].shape[0]*100.0/df.shape[0]))
            if f1_ == 'notWeighted':
                f1 = f1_score(acts, preds, average='macro')
            if f1_ == 'weighted':
                f1 = f1_score(acts, preds, average='weighted')
            print('Unseen test set {} f1_score: {}'.format(f1_, f1)) 
            self.show_cf(preds, acts, cf_type)
        elif self.modelType == 'pair':
            print('Evaluation beginning')
            n = len(self.unseen_dl.dataset.subset)
            m = self.train_sample_size
            self.distMatrix = torch.empty((n, int(m/2)))
            self.predLabels = torch.empty((n, 1))
            self.actLabels = torch.empty((0, 1))
            self.trainPreds = torch.empty((0, self.outputSize))
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            d = nn.PairwiseDistance(p=2)
            #euclidean_distance = cos(features_1, features_2)
            self.model.eval()
            for i, batch in enumerate(self.unseen_dl):
                with torch.no_grad():
                    testPreds = self.model(batch['image'].cuda())
                    #actLabels = torch.argmax(batch['labels'], 1).float().reshape(-1,1))
                    self.actLabels = torch.cat((self.actLabels, torch.argmax(batch['labels'], 1).float().reshape(-1,1)), 0)
            for j in range(0, m, self.BATCH_SIZE):
                with torch.no_grad():
                    if self.mining == 'random':
                        preds = self.model(self.train_img_1[j: j + self.BATCH_SIZE].cuda())
                    if self.mining == 'hard':
                        preds = self.model(self.train_hard_img_1[j: j + self.BATCH_SIZE].cuda())
                    for ii in range(0, preds.shape[0], 2):
                        self.trainPreds = torch.cat((self.trainPreds, preds[ii].reshape(1,-1).cpu()), 0)
            for row in range(0, n):
                for col in range(0, int(m/2)):
                    self.distMatrix[row, col] = d(testPreds[row].reshape(1,-1).cuda(),
                                                               self.trainPreds[col].reshape(1,-1).cuda()).cpu()
                    torch.cuda.empty_cache()
            for r in range(0, n):
                ind = torch.argmin(self.distMatrix[r])
                if self.mining == 'random':
                    self.predLabels[r, 0] = self.train_label_1[ind*2]
                if self.mining == 'hard':
                    self.predLabels[r, 0] = self.train_hard_label_1[ind*2]
            
            preds = self.predLabels.numpy()
            acts = self.actLabels.numpy()
            df = pd.DataFrame(preds, columns=['predict'])
            df['actuals'] = acts
            print('Unseen test set (contrastive) accuracy: {}'.format(df[df['predict'] == df['actuals']].shape[0]*100.0/df.shape[0]))
            if f1_ == 'notWeighted':
                f1 = f1_score(acts, preds, average='macro')
            if f1_ == 'weighted':
                f1 = f1_score(acts, preds, average='weighted')
            print('Unseen test set {} f1_score: {}'.format(f1_, f1)) 
            self.show_cf(preds, acts, cf_type)
    
    def evaluate2(self, f1_='weighted', cf_type='numbers'):
        if self.modelType == 'triplet':
            print('Evaluation beginning')
            n = len(self.unseen_dl.dataset.subset)
            unique_labs = torch.unique(self.train_label_anchor)
            m = self.train_sample_size
            n_classes = len(unique_labs)
            self.avg_fts = torch.empty((0, self.outputSize))
            self.distMatrix = torch.empty((n, n_classes))
            self.predLabels = torch.empty((n, 1))
            self.actLabels = torch.empty((0, 1))
            self.trainPreds = torch.empty((0, self.outputSize))
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            d = nn.PairwiseDistance(p=2)
            self.model.eval()
            for i, batch in enumerate(self.unseen_dl):
                with torch.no_grad():
                    self.testPreds = self.model(batch['image'].cuda())
                    self.actLabels = torch.cat((self.actLabels, torch.argmax(batch['labels'], 1).float().reshape(-1,1)), 0)
            for j in range(0, m, self.BATCH_SIZE):
                with torch.no_grad():
                    if self.mining == 'random':
                        preds = self.model(self.train_img_anchor[j: j + self.BATCH_SIZE].cuda())
                    if self.mining == 'hard':
                        preds = self.model(self.train_hard_anchors_img[j: j + self.BATCH_SIZE].cuda())
                    self.trainPreds = torch.cat((self.trainPreds, preds.cpu()), 0)
            
            for lab in unique_labs:
                if self.mining == 'random':
                    inds = np.where(self.train_label_anchor == lab)[0]
                if self.mining == 'hard':
                    inds = np.where(self.train_hard_anchors_labels == lab.float())[0]
                fts = torch.empty((0, self.outputSize))
                for k in range(0, inds.shape[0], self.BATCH_SIZE):
                    with torch.no_grad():
                        #preds = model(train_img_anchor[inds[j: j + BATCH_SIZE]].cuda())
                        predss = self.trainPreds[inds[k: k + self.BATCH_SIZE]]
                        fts = torch.cat((fts, predss.cpu()), 0)
                avg = torch.mean(fts, 0).reshape(1,-1)
                self.avg_fts = torch.cat((self.avg_fts, avg ), 0)
            
            for row in range(0, n):
                for col in range(0, n_classes):
                    self.distMatrix[row, col] = d(self.testPreds[row].reshape(1,-1).cuda(),
                                                               self.avg_fts[col].reshape(1,-1).cuda()).cpu()
                    #torch.cuda.empty_cache()
            for r in range(0, n):
                self.predLabels[r, 0] = torch.argmin(self.distMatrix[r])
                
            preds = self.predLabels.numpy()
            acts = self.actLabels.numpy()
            df = pd.DataFrame(preds, columns=['predict'])
            df['actuals'] = acts
            print('Unseen test set accuracy: {}'.format(df[df['predict'] == df['actuals']].shape[0]*100.0/df.shape[0]))
            if f1_ == 'notWeighted':
                f1 = f1_score(acts, preds, average='macro')
            if f1_ == 'weighted':
                f1 = f1_score(acts, preds, average='weighted')
            print('Unseen test set {} f1_score: {}'.format(f1_, f1)) 
            self.show_cf(preds, acts, cf_type)
        elif self.modelType == 'pair':
            print('Evaluation beginning')
            n = len(self.unseen_dl.dataset.subset)
            m = self.train_sample_size
            self.predLabels = torch.empty((n, 1))
            self.actLabels = torch.empty((0, 1))
            self.trainPreds = torch.empty((0, self.outputSize))
            if self.mining == 'random':
                self.unique_labs = torch.unique(self.train_label_1)
            if self.mining == 'hard':
                self.unique_labs = torch.unique(self.train_hard_label_1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            self.trainLabels = torch.empty((0, 1))
            
            if self.mining == 'random':
                for jj in range(0, self.train_label_1.shape[0], 2):
                    self.trainLabels = torch.cat((self.trainLabels, self.train_label_1[jj].reshape(1,-1).float()), 0)
            if self.mining == 'hard':
                for jj in range(0, self.train_hard_label_1.shape[0], 2):
                    self.trainLabels = torch.cat((self.trainLabels, self.train_hard_label_1[jj].reshape(1,-1)), 0)
            self.avg_fts = torch.empty((0, self.outputSize))
            self.distMatrix = torch.empty((n,len(self.unique_labs)))
            d = nn.PairwiseDistance(p=2)
            #euclidean_distance = cos(features_1, features_2)
            self.model.eval()
            for i, batch in enumerate(self.unseen_dl):
                with torch.no_grad():
                    testPreds = self.model(batch['image'].cuda())
                    #actLabels = torch.argmax(batch['labels'], 1).float().reshape(-1,1))
                    self.actLabels = torch.cat((self.actLabels, torch.argmax(batch['labels'], 1).float().reshape(-1,1)), 0)
            for j in range(0, m, self.BATCH_SIZE):
                with torch.no_grad():
                    if self.mining == 'random':
                        preds = self.model(self.train_img_1[j: j + self.BATCH_SIZE].cuda())
                    if self.mining == 'hard':
                        preds = self.model(self.train_hard_img_1[j: j + self.BATCH_SIZE].cuda())
                    for ii in range(0, preds.shape[0], 2):
                        self.trainPreds = torch.cat((self.trainPreds, preds[ii].reshape(1,-1).cpu()), 0)
            
            for lab in self.unique_labs:
                inds = np.where(self.trainLabels == lab.float())[0]
                fts = torch.empty((0, self.outputSize))
                for k in range(0, inds.shape[0], self.BATCH_SIZE):
                    with torch.no_grad():
                        #preds = model(train_img_anchor[inds[j: j + BATCH_SIZE]].cuda())
                        predss = self.trainPreds[inds[k: k + self.BATCH_SIZE]]
                        fts = torch.cat((fts, predss.cpu()), 0)
                avg = torch.mean(fts, 0).reshape(1,-1)
                self.avg_fts = torch.cat((self.avg_fts, avg ), 0)
            
            for row in range(0, n):
                for col in range(0, len(self.unique_labs)):
                    self.distMatrix[row, col] = d(testPreds[row].reshape(1,-1).cuda(),
                                                               self.avg_fts[col].reshape(1,-1).cuda()).cpu()
                    torch.cuda.empty_cache()
            for r in range(0, n):
                ind = torch.argmin(self.distMatrix[r])
                if self.mining =='random':
                    self.predLabels[r, 0] = ind
                if self.mining =='hard':
                    self.predLabels[r, 0] = ind
            
            preds = self.predLabels.numpy()
            acts = self.actLabels.numpy()
            df = pd.DataFrame(preds, columns=['predict'])
            df['actuals'] = acts
            print('Unseen test set (contrastive) accuracy: {}'.format(df[df['predict'] == df['actuals']].shape[0]*100.0/df.shape[0]))
            if f1_ == 'notWeighted':
                f1 = f1_score(acts, preds, average='macro')
            if f1_ == 'weighted':
                f1 = f1_score(acts, preds, average='weighted')
            print('Unseen test set {} f1_score: {}'.format(f1_, f1)) 
            self.show_cf(preds, acts, cf_type)


      
    
    
