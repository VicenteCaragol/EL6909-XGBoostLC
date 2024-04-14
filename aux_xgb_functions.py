# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:50:39 2024

@author: vicen
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve,precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score



def LabelEncoder(y_data):
    nclass = len(np.unique(y_data))
    LabelEncoder_dict = {
    4:{'SNIa':0,'SNII':1,'SNIbc':2,'SLSN':3},
    5:{'QSO':0,'AGN':1, 'YSO':2,'Blazar':3, 'CV/Nova':4},
    6:{'E':0,'RRL':1,'LPV':2,'Periodic-Other':3,'DSCT':4,'CEP':5},
    3:{'Transient':0, 'Stochastic':1, 'Periodic':2},
    15:{'SNIa':0,'SNII':1,'SNIbc':2,'SLSN':3,'QSO':4,'AGN':5, 'YSO':6,'Blazar':7, 'CV/Nova':8,'E':9,'RRL':10,'LPV':11,'Periodic-Other':12,'DSCT':13,'CEP':14}}
    y_data_encoded = y_data.apply(lambda x: LabelEncoder_dict[nclass][x])
    return y_data_encoded


def recall_mod_v1(y_true,y_pred,weights):
    weights=weights/np.sum(weights)

    return np.dot(weights.sort_index(),recall_score(y_true, y_pred, average=None))

def XGB_crossval(X_sc,y_sc, n_splits=10,params=None,mod=False):
    y_sc_enc= LabelEncoder(y_sc)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.33, random_state=42)
    i=0
    n_classes=len(y_sc_enc.unique())
    precisions=np.zeros(n_splits)
    recalls=np.zeros(n_splits)
    modrecalls=np.zeros(n_splits)
    minrecalls=np.zeros(n_splits)
    f1s=np.zeros(n_splits)
    rocaucs=np.zeros(n_splits)
    matrixes=np.zeros((n_splits,n_classes,n_classes))
    matrixesn=np.zeros((n_splits,n_classes,n_classes))
    
    for train_index, test_index in sss.split(X_sc, y_sc_enc):
        print(f'Validation {i+1}/{n_splits}')
        X_train, X_valid = X_sc.iloc[train_index], X_sc.iloc[test_index]
        y_train, y_valid = y_sc_enc.iloc[train_index], y_sc_enc.iloc[test_index]
        
    
        ###Step 1
        class_weights = list(class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train))
        dtrain = xgb.DMatrix(data=X_train, label=y_train
                                 ,weight=pd.Series(y_train).map(lambda x: class_weights[x])
                                )
        dvalid=xgb.DMatrix(data=X_valid, label=y_valid)
        if params==None:
            params={}
            params['eta']=0.025
            params['min_child_weight']=6
            params['subsample']=0.9
            params['gamma']=1.5
            params['colsample_bytree']=0.95
            params['max_delta_step']=5
            params['max_depth'] = 4
            
        params['eval_metric']= 'mlogloss'
        params['objective']= 'multi:softprob'
        params['booster']= 'gbtree'
        params['tree_method']= 'hist'
        params['device']= 'cuda'
        params['num_class']= n_classes
        watchlist = [(dtrain, 'train')]
        
        xgb_model = xgb.train(params, dtrain, 500,
                                evals=watchlist,
                                verbose_eval=False,
                                early_stopping_rounds=15,
                                balanced_bootstrap=mod)
    
        preds = xgb_model.predict(dvalid, iteration_range=(1,xgb_model.best_iteration + 1))
    
        for j in range(len(preds)):
            preds[j] = np.exp(preds[j])/np.sum(np.exp(preds[j]))       
        y_pred = np.array(np.argmax(np.array(preds),axis=1), ndmin=1)
    
        precisions[i]=precision_score(y_valid,y_pred,average='macro')
        recalls[i]=recall_score(y_valid,y_pred,average='macro')
        f1s[i]=f1_score(y_valid,y_pred,average='macro')
        matrixes[i]=confusion_matrix(y_valid,y_pred)
        matrixesn[i]=confusion_matrix(y_valid,y_pred,normalize='true')
        minrecalls[i]=np.min(recall_score(y_valid, y_pred, average=None))
        modrecalls[i]=recall_mod_v1(y_valid, y_pred, 1/LabelEncoder(y_sc).value_counts(normalize=True))
    
        i=i+1
    
    
    
    print(f'Precision: Mean = {np.mean(precisions):.2f}  Std = {np.std(precisions):.2f}')
    print(f'Recall: Mean = {np.mean(recalls):.2f}  Std = {np.std(recalls):.2f}')
    print(f'F1-score: Mean = {np.mean(f1s):.2f}  Std = {np.std(f1s):.2f}')
    print(f'Mod. Recall: Mean = {np.mean(modrecalls):.2f}  Std = {np.std(modrecalls):.2f}')
    print(f'Min. Recall: Mean = {np.mean(minrecalls):.2f}  Std = {np.std(minrecalls):.2f}')
    
    disp=ConfusionMatrixDisplay(np.mean(matrixes,axis=0)#, display_labels=Labels_dict[subclass]
                               )
    disp.plot(cmap='Blues',values_format='.1f',colorbar=False)
    plt.title('Matriz de confusión (Cantidad de elementos)')
    
    disp=ConfusionMatrixDisplay(np.mean(matrixesn,axis=0)#, display_labels=Labels_dict[subclass]
                               )
    disp.plot(cmap='Blues',values_format='.2%',colorbar=False)
    plt.title('Matriz de confusión (Porcentaje)')
    plt.show()
    
from imblearn.ensemble import BalancedRandomForestClassifier
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def BRF_crossval(X_sc,y_sc):
    n_splits=10
    y_sc_enc= LabelEncoder(y_sc)
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.33, random_state=42)
    i=0
    n_classes=len(y_sc.unique())
    precisions=np.zeros(n_splits)
    recalls=np.zeros(n_splits)
    f1s=np.zeros(n_splits)
    matrixes=np.zeros((n_splits,n_classes,n_classes))
    matrixesn=np.zeros((n_splits,n_classes,n_classes))
    minrecalls=np.zeros(n_splits)
    modrecalls=np.zeros(n_splits)
    for train_index, test_index in sss.split(X_sc, y_sc_enc):
        print(f'Validation {i+1}/{n_splits}')
        X_train, X_test = X_sc.iloc[train_index], X_sc.iloc[test_index]
        y_train, y_test = y_sc_enc.iloc[train_index], y_sc_enc.iloc[test_index]
        
        X_train_original = X_train.copy()
        X_train_original[X_train_original > 1e32] = 0
        X_train = X_train_original
        
        X_train=X_train.fillna(-999)
        X_test=X_test.fillna(-999)
        
        
    
        rf_model = BalancedRandomForestClassifier(
                             n_estimators=500,
                             max_features=13,#'auto',
                             max_depth=None,
                             n_jobs=-1,
                             bootstrap=True,
                             criterion='gini',
                             min_samples_split=10,
                             min_samples_leaf=10)
        rf_model.fit(X_train,y_train)
        preds_test = rf_model.predict(X_test)
        
        precisions[i]=precision_score(y_test,preds_test,average='macro')
        recalls[i]=recall_score(y_test,preds_test,average='macro')
        f1s[i]=f1_score(y_test,preds_test,average='macro')
        matrixes[i]=confusion_matrix(y_test,preds_test)
        matrixesn[i]=confusion_matrix(y_test,preds_test,normalize='true')
        minrecalls[i]=np.min(recall_score(y_test, preds_test, average=None))
        modrecalls[i]=recall_mod_v1(y_test,preds_test, 1/LabelEncoder(y_sc).value_counts(normalize=True))
        i=i+1
    
    
    
    print(f'Precision: Mean = {np.mean(precisions):.2f}  Std = {np.std(precisions):.2f}')
    print(f'Recall: Mean = {np.mean(recalls):.2f}  Std = {np.std(recalls):.2f}')
    print(f'F1-score: Mean = {np.mean(f1s):.2f}  Std = {np.std(f1s):.2f}')
    print(f'Mod. Recall: Mean = {np.mean(modrecalls):.2f}  Std = {np.std(modrecalls):.2f}')
    print(f'Min. Recall: Mean = {np.mean(minrecalls):.2f}  Std = {np.std(minrecalls):.2f}')
    
    disp=ConfusionMatrixDisplay(np.mean(matrixes,axis=0)#, display_labels=Labels_dict[subclass]
                               )
    disp.plot(cmap='Blues',values_format='.1f',colorbar=False)
    plt.title('Matriz de confusión (Cantidad de elementos)')
    
    disp=ConfusionMatrixDisplay(np.mean(matrixesn,axis=0)#, display_labels=Labels_dict[subclass]
                               )
    disp.plot(cmap='Blues',values_format='.2%',colorbar=False)
    plt.title('Matriz de confusión (Porcentaje)')
    plt.show()