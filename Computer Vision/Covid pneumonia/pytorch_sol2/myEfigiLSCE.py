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

from pytorch_metric_learning import losses, miners, distances, reducers, testers, regularizers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

class myEfigiLSCE:
    def __init__(self,train_dl, val_dl, unseen_dl, model, optimizer, scheduler, criterion, savePath='./models/efigi_label_smoothin_128', device='cuda', BATCH_SIZE=64):
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.unseen_dl = unseen_dl
        self.BATCH_SIZE = BATCH_SIZE
        self.savePath = savePath
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
    
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
            self.optimizer.zero_grad()             # 1. Zero the parameter gradients
            pred = nn.Softmax(dim=1)
            outputs = self.model(inputs) 
            loss = self.criterion(outputs, torch.max(labels, 1)[1])
            losses.append(loss.item())
            loss.backward()           
            self.optimizer.step()
            del inputs
            del labels
            del outputs
            torch.cuda.empty_cache()
        epoch_loss = sum(losses)/len(losses)
        return epoch_loss
    
    def unseen_phase(self):
        self.model.eval()
        losses = []
        preds = torch.empty(0, 9).long().to(self.device)
        acts = torch.empty(0, 9).long().to(self.device)
        for i, batch in enumerate(dl):
            with torch.no_grad():
                inputs, labels = batch['image'], batch['labels'].long().view(-1,9)
                inputs, labels = inputs.to(device), labels.to(device)
                pred = nn.Softmax(dim=1)
                outputs = self.model(inputs)  
                outputs = pred(outputs)
                preds = torch.cat((preds, outputs), 0)
                acts = torch.cat((acts, labels.long()), 0)
                for j in range(0,batch['labels'].size()[0]):
                    if torch.max(outputs, 1)[1][j] == torch.max(labels, 1)[1][j]:
                        losses.append(1)
                    else:
                        losses.append(0)
            del inputs
            del labels
            del batch
            torch.cuda.empty_cache()
        epoch_loss = (sum(losses)*100.00)/len(losses)
        return epoch_loss, preds, acts


    def train(self, n_epochs=5):
        #self.model = self.model.to(self.device)
        train_losses = []
        val_losses= []
        #pth = './efigiCheckpoints/efigicheckpoint'
        #early_stopping = EarlyStopping(patience=10, verbose=True, path=pth)
        print("Training beginning")
        for epoch in range(n_epochs):
            train_loss = self.train_phase(tr='train')
            train_losses.append(train_loss)
            print("Epoch: {} Train Loss: {}".format(epoch+1, train_loss))
           
            val_loss = self.train_phase(tr='val')
            self.scheduler.step(val_loss)       
            val_losses.append(val_loss)
            print("Epoch: {} Val Loss: {}".format(epoch+1, val_loss))
        
        pickle.dump(train_losses,open('./losses/efigi_lsce_train','wb'))
        pickle.dump(val_losses,open('./losses/efigi_lsce_val','wb'))
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
        
            