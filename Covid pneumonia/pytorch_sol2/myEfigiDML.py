#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 14:18:40 2020

@author: mohamed
"""

from __future__ import print_function, division
import matplotlib.pyplot as plt
import torch
import pickle
from pytorchtools import EarlyStopping
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler

from pytorch_metric_learning import losses, miners, distances, reducers, testers, regularizers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class myEfigiDML:
    def __init__(self,train_dl, val_dl, unseen_dl, model, optimizer, scheduler, criterion, mining_function, loss, savePath='./models/', device='cuda', BATCH_SIZE=64):
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.unseen_dl = unseen_dl
        self.BATCH_SIZE = BATCH_SIZE
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.mining_function = mining_function
        self.loss = loss
        self.distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        self.reducer = reducers.ThresholdReducer(low = 0)
        self.regularizer = regularizers.LpRegularizer(p=2)
        if self.mining_function == 'triplet':
            self.mining_func = miners.TripletMarginMiner(margin = 0.01, distance = self.distance,
                                            type_of_triplets = "semihard")
        elif self.mining_function == 'pair':
            self.mining_func = miners.PairMarginMiner(pos_margin=0, neg_margin=0.2)
        
        if self.loss == 'triplet':
            self.loss_function = losses.TripletMarginLoss(margin = 0.01, distance = self.distance, reducer = self.reducer)
        elif self.loss == 'contrastive':
            self.loss_function = losses.ContrastiveLoss(pos_margin=0, neg_margin=1.5)
        elif self.loss == 'panc':
            self.loss_function = losses.ProxyAnchorLoss(9,128, margin=0.01, alpha=5, reducer=self.reducer,
                                   weight_regularizer = self.regularizer)
        elif self.loss == 'pnca':
            self.loss_function = losses.ProxyNCALoss(9, 128, softmax_scale=1, reducer = self.reducer,            weight_regularizer=self.regularizer)
        elif self.loss == 'normsoftmax':
            self.loss_function = losses.NormalizedSoftmaxLoss(9, 128, temperature=0.05, reducer=self.reducer,                                        weight_regularizer=self.regularizer)
            
            
        if self.loss in ['normsoftmax', 'panc', 'pnca']:
            self.loss_optimizer = optim.SGD(self.loss_function.parameters(), lr=0.0001, momentum=0.9)
            self.loss_scheduler = lr_scheduler.ReduceLROnPlateau(self.loss_optimizer, 'min',patience=3, 
                                            threshold=0.0001,factor=0.1, verbose=True)
        
        self.savePath = savePath + 'efigi{}_{}_128'.format(self.mining_function, self.loss)
    
    def train_phase(self, tr='train'):
        if tr == 'train':
            self.model.train()
            dl = self.train_dl
        if tr == 'val':
            self.model.eval()
            dl = self.val_dl
        losses = []
        for i, batch in enumerate(dl):
            inputs, labels = batch['image'], batch['labels'].long().view(-1,9)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = torch.argmax(labels, 1)
            self.optimizer.zero_grad()             
            if self.loss in ['normsoftmax', 'panc', 'pnca']:
                self.loss_optimizer.zero_grad()
            embeddings = self.model(inputs)           
            indices_tuple = self.mining_func(embeddings, labels)
            eloss = self.loss_function(embeddings, labels, indices_tuple)
            losses.append(eloss.item())
            eloss.backward()           
            if self.loss in ['normsoftmax', 'panc', 'pnca']:
                self.loss_optimizer.step()
            self.optimizer.step()
            del inputs
            del labels
            del embeddings
            torch.cuda.empty_cache()
        epoch_loss = sum(losses)/len(losses)
        return epoch_loss
    
    def train(self, n_epochs=5):
        #self.model = self.model.to(self.device)
        train_losses = []
        val_losses= []
        print("Training beginning")
        for epoch in range(n_epochs):
            train_loss = self.train_phase(tr='train')
            train_losses.append(train_loss)
            print("Epoch: {} Train Loss: {}".format(epoch+1, train_loss))
           
            val_loss = self.train_phase(tr='val')
            self.scheduler.step(val_loss)       
            val_losses.append(val_loss)
            print("Epoch: {} Val Loss: {}".format(epoch+1, val_loss))
        
        #pickle.dump(train_losses,open('./losses/efigi_lsce_train','wb'))
        #pickle.dump(val_losses,open('./losses/efigi_lsce_val','wb'))
        torch.save(self.model, self.savePath)
        print('Model saved: ' + self.savePath)
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()
        print("Training complete") 
        return None
    
    def get_all_embeddings(self, dl):
        labs = torch.empty((0, 1)).to(self.device)
        embeds = torch.empty((0,128)).to(self.device)
        for i, batch in enumerate(dl):
            with torch.no_grad():
                inputs, labels = batch['image'], batch['labels'].float().view(-1,9)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = torch.argmax(labels, 1)
                embeds = torch.cat((embeds, self.model(inputs)), 0)
                labs = torch.cat((labs, labels.reshape(-1, 1)), 0)
                del inputs
                del labels
                torch.cuda.empty_cache()
        return embeds, labs

    def RSAtest(self, test_dl):
        print('Evaluation on RSA test set beginning:')
        trEmbeds, trLabels = self.get_all_embeddings(self.train_dl)
        rEmbeds, rLabels = self.get_all_embeddings(test_dl)
        accuracy_calculator = AccuracyCalculator(include = (), k = 10)
        results = accuracy_calculator.get_accuracy(rEmbeds.cpu().numpy(), 
                                                trEmbeds.cpu().numpy(),
                                                rLabels.reshape((-1)).cpu().numpy(),
                                                trLabels.reshape((-1)).cpu().numpy(),
                                False)
        print('Accuracy on RSA test set: {}'.format(round(results['precision_at_1']*100.00,2)))
        print('MAP@R on RSA test set: {}'.format(round(results['mean_average_precision'], 4)))
        return results
        
            