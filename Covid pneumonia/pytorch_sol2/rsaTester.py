#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:44:48 2020

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

class RSA_tester:
    def __init__(self,dl, catalog='efigi', device='cuda', BATCH_SIZE=64):
        self.device = device
        self.dl = dl
        self.catalog = catalog
        if self.catalog == 'efigi':
            self.model = torch.load('./models/efigi')
            self.nClasses = 9
        elif self.catalog == 'gz2':
            self.model = torch.load('./models/gz2_hubble')
            self.nClasses = 11
        self.model = self.model.to(self.device)
        
    def test_phase(self):
        self.model.eval()
        losses = []
        preds = torch.empty(0, self.nClasses).to(self.device)
        acts = torch.empty(0, 11).to(self.device)
        for i, batch in enumerate(self.dl):
            with torch.no_grad():
                inputs, labels = batch['image'], batch['labels'].long().view(-1,11)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                pred = nn.Softmax(dim=1)
                outputs = self.model(inputs)  
                outputs = pred(outputs)
                preds = torch.cat((preds, outputs), 0)
                acts = torch.cat((acts, labels.float()), 0)
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
    
    def efigi(self, f1_='weighted', cf_='numbers'):
        test_acc, self.preds, self.acts = self.test_phase()
        efigi = pd.read_csv('./t/efigi_model.csv')
        efigi['codes'] = efigi['hubb'].astype('category').cat.codes
        efigi = efigi.loc[:, ['hubb','codes']]
        efigi = efigi.drop_duplicates(subset=['hubb'])
        preds_arr = self.preds.cpu().numpy()
        args = np.zeros((preds_arr.shape[0], 1))
        for i in range(preds_arr.shape[0]):
            args[i, 0] = np.argmax(preds_arr[i])
        act_arr = self.acts.cpu().numpy()
        args_act = np.zeros((act_arr.shape[0], 1))
        for i in range(act_arr.shape[0]):
            args_act[i, 0] = np.argmax(act_arr[i])
        df = pd.DataFrame(args, columns=['Predict'])
        df['Act'] = args_act
        orig_df = self.dl.dataset.classes_frame
        orig_df['cat'] = 0
        for i in range(orig_df.shape[0]):
            orig_df.loc[i,'cat'] = np.argmax(orig_df.iloc[i,2:].values)
        orig_df = orig_df.loc[:, ['New type', 'cat']].drop_duplicates()
        sdf = pd.merge(df, orig_df, left_on=['Act'], right_on=['cat'], how='left')
        sdf = pd.merge(sdf, efigi, left_on=['Predict'], right_on=['codes'], how='left')
        
        actuals = sdf.loc[:,['New type', 'cat']]
        actuals['cat'].value_counts()
        change = actuals[actuals['New type'].isin(['E0','E3-5','E7'])].index.tolist()
        actuals.loc[change, 'New type'] = 'E'
        del actuals['cat']
        actuals['code'] = actuals['New type'].astype('category').cat.codes
        
        predicts = sdf.loc[:, ['hubb', 'codes']]
        change = predicts[predicts['hubb'].isin(['E0','E3-5','E7'])].index.tolist()
        predicts.loc[change, 'hubb'] = 'E'
        del predicts['codes']
        predicts['code'] = predicts['hubb'].astype('category').cat.codes
        predicts['hubb'].value_counts()
        
        
        change = sdf[sdf['New type'].isin(['E0','E3-5','E7'])].index.tolist()
        sdf.loc[change, 'New type'] = 'E'
        change = sdf[sdf['hubb'].isin(['E0','E3-5','E7'])].index.tolist()
        sdf.loc[change, 'hubb'] = 'E'
        acc = (sdf[sdf['New type'] == sdf['hubb']].shape[0]*100.00)/sdf.shape[0]
        print('RSA accuracy: {}'.format(acc))
        if f1_ == 'notWeighted':
            f1 = f1_score(sdf['New type'], sdf['hubb'], average='macro')
        if f1_ == 'weighted':
            f1 = f1_score(sdf['New type'], sdf['hubb'], average='weighted')
        print('RSA ' + f1_ + ' f1 score: {}'.format(f1))
        cf = confusion_matrix(sdf['New type'].tolist(), sdf['hubb'].tolist(), labels=['E','Irr','S0','SBa','SBb','SBc','Sa','Sb','Sc'])
        C = cf / cf.astype(np.float).sum(axis=1, keepdims=True)
        if cf_ == 'numbers':
            print("Confusion matrix:\n%s" % cf)
        if cf_ == 'percentage':
            print("Confusion matrix:\n%s" % C)
    
    def gz2(self, f1_='weighted', cf_='numbers'):
        test_acc, self.preds, self.acts = self.test_phase()

        preds_arr = self.preds.cpu().numpy()
        args = np.zeros((preds_arr.shape[0], 1))
        for i in range(preds_arr.shape[0]):
            args[i, 0] = np.argmax(preds_arr[i])
        act_arr = self.acts.cpu().numpy()
        args_act = np.zeros((act_arr.shape[0], 1))
        for i in range(act_arr.shape[0]):
            args_act[i, 0] = np.argmax(act_arr[i])
        df = pd.DataFrame(args, columns=['Predict'])
        df['Act'] = args_act
        orig_df = self.dl.dataset.classes_frame
        orig_df['cat'] = 0
        for i in range(orig_df.shape[0]):
            orig_df.loc[i,'cat'] = np.argmax(orig_df.iloc[i,2:].values)
        orig_df = orig_df.loc[:, ['New type', 'cat']].drop_duplicates()
        sdf = pd.merge(df, orig_df, left_on=['Act'], right_on=['cat'], how='left')
        sdf = pd.merge(sdf, orig_df, left_on=['Predict'], right_on=['cat'], how='left')
        
        actuals = sdf.loc[:,['New type_x', 'cat_x']]
        actuals['cat_x'].value_counts()
        change = actuals[actuals['New type_x'].isin(['E0','E3-5','E7'])].index.tolist()
        actuals.loc[change, 'New type_x'] = 'E'
        del actuals['cat_x']
        actuals['code'] = actuals['New type_x'].astype('category').cat.codes
        
        predicts = sdf.loc[:, ['New type_y', 'cat_y']]
        change = predicts[predicts['New type_y'].isin(['E0','E3-5','E7'])].index.tolist()
        predicts.loc[change, 'New type_y'] = 'E'
        del predicts['cat_y']
        predicts['code'] = predicts['New type_y'].astype('category').cat.codes
        
        
        acc = (sdf[sdf['New type_x'] == sdf['New type_y']].shape[0]*100.00)/sdf.shape[0]
        print('RSA accuracy: {}'.format(acc))
        if f1_ == 'notWeighted':
            f1 = f1_score(actuals['New type_x'], predicts['New type_y'], average='macro')
        if f1_ == 'weighted':
            f1 = f1_score(actuals['New type_x'], predicts['New type_y'], average='weighted')
        print('RSA ' + f1_ + ' f1 score: {}'.format(f1))
        cf = confusion_matrix(sdf['New type_x'].tolist(), sdf['New type_y'].tolist())
        C = cf / cf.astype(np.float).sum(axis=1, keepdims=True)
        if cf_ == 'numbers':
            print("Confusion matrix:\n%s" % cf)
        if cf_ == 'percentage':
            print("Confusion matrix:\n%s" % C)
        
    def evaluate(self, f1_='weighted', cf_='numbers'):
        if self.catalog == 'efigi':
            self.efigi(f1_, cf_)
        if self.catalog == 'gz2':
            self.gz2(f1_, cf_)
            
            
        
        
        