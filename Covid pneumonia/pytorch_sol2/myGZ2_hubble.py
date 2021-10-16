#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 13:03:41 2020

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
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class GZ2_hubble:
    def __init__(self,train_dl, val_dl, unseen_dl, model, optimizer, scheduler, criterion, savePath='./models/gz2_hubble', device='cuda', BATCH_SIZE=64):
        self.device = device
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.unseen_dl = unseen_dl
        self.BATCH_SIZE = BATCH_SIZE
        self.savePath = savePath
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        
    
        
    def change_model(self):
        self.model.fc[3] = nn.Linear(512, 11)
        del self.model.fc[4]
        return None
     
    def train_phase(self, tr='train'):
        if tr == 'train':
            self.model.train()
            dl = self.train_dl
        if tr == 'val':
            self.model.eval()
            dl = self.val_dl            
        losses = []
        for i, batch in enumerate(dl):
            inputs, labels = batch['image'], batch['labels'].long().view(-1,11)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()             # 1. Zero the parameter gradients
            outputs = self.model(inputs)           # 2. Run the model
    
            loss = self.criterion(outputs, torch.max(labels, 1)[1]) # 3. Calculate loss
            losses.append(loss.item())
            loss = torch.sqrt(loss)           #    -> RMSE loss
            loss.backward()                   # 4. Backward propagate the loss
            self.optimizer.step()                  # 5. Optimize the network
 
            del batch
            del inputs
            del labels
            torch.cuda.empty_cache()
        epoch_loss = sum(losses) / len(losses)
        return epoch_loss
    
    def unseen_phase(self):
        self.model.eval()
        losses = []
        preds = torch.empty(0, 11).to(self.device)
        acts = torch.empty(0, 11).to(self.device)
        for i, batch in enumerate(self.unseen_dl):
            with torch.no_grad():
                inputs, labels = batch['image'], batch['labels'].long().view(-1,11)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                pred = nn.Softmax(dim=1)
                outputs = self.model(inputs)  
                outputs = pred(outputs)
                preds = torch.cat((preds, outputs), 0)
                acts = torch.cat((acts, labels.float()), 0)
                #loss = criterion(outputs, torch.max(labels, 1)[1]) # 3. Calculate loss
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
        self.change_model()
        self.model = self.model.to(self.device)
        train_losses = []
        val_losses= []
        pth = './gz2hub_checkpoints/gz2hubcheckpoint'
        early_stopping = EarlyStopping(patience=10, verbose=True, path=pth)
        print('Training beginning')
        for epoch in range(n_epochs):
            train_loss = self.train_phase(tr='train')
            train_losses.append(train_loss)
            print("Epoch: {} Train Loss: {}".format(epoch+1, train_loss))
           
            val_loss = self.train_phase(tr='val')
            self.scheduler.step(val_loss)       
            val_losses.append(val_loss)
            print("Epoch: {} Val Loss: {}".format(epoch+1, val_loss))
            
            early_stopping(val_loss, self.model, epoch)
            if early_stopping.early_stop:
                ep = epoch - 10
                self.model.load_state_dict(torch.load('./gz2hub_checkpoints/gz2hubcheckpoint{}.pt'.format(ep)))
                print("Early stopping")
                break
        
        pickle.dump(train_losses,open('./losses/gz2hub_train','wb'))
        pickle.dump(val_losses,open('./losses/gz2hub_val','wb'))
        torch.save(self.model, self.savePath)
        print('Model saved: ' + self.savePath)
        plt.plot(train_losses, label='Training loss')
        plt.plot(val_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.show()
        print("Training complete") 
        return None
    
    def show_cf(self, f1_='weighted', cf_type='numbers'):
        print('Evaluation on unseen test set beginning:')
        unseen_loss, preds, acts = self.unseen_phase()
        preds_arr = preds.cpu().numpy()
        args = np.zeros((preds_arr.shape[0], 1))
        for i in range(preds_arr.shape[0]):
            args[i, 0] = np.argmax(preds_arr[i])
        
        act_arr = acts.cpu().numpy()
        args_act = np.zeros((act_arr.shape[0], 1))
        for i in range(act_arr.shape[0]):
            args_act[i, 0] = np.argmax(act_arr[i])    
        
        df = pd.DataFrame(args, columns=['Predict'])
        df['Act'] = args_act
        orig_df = self.unseen_dl.dataset.classes_frame
        orig_df['cat'] = 0
        for i in range(orig_df.shape[0]):
            orig_df.loc[i,'cat'] = np.argmax(orig_df.iloc[i,5:].values)
        orig_df = orig_df.loc[:, ['hubble_type', 'cat']].drop_duplicates()
        df = pd.merge(df, orig_df, left_on=['Act'], right_on=['cat'], how='left')
        df = pd.merge(df, orig_df, left_on=['Predict'], right_on=['cat'], how='left')
        cf = confusion_matrix(df['hubble_type_x'].tolist(), df['hubble_type_y'].tolist())
        C = cf / cf.astype(np.float).sum(axis=1, keepdims=True)
        print('Unseen accuracy: {}'.format(unseen_loss))
        if f1_ == 'notWeighted':
            f1 = f1_score(args, args_act, average='macro')
        if f1_ == 'weighted':
            f1 = f1_score(args, args_act, average='weighted')
        print('Unseen ' + f1_ + ' f1 score: {}'.format(f1))
        if cf_type == 'numbers':
            print("Confusion matrix:\n%s" % cf)
        if cf_type == 'percentages':
            print("Confusion matrix:\n%s" % C)
            
        
        
        
        
        