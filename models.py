import os
import gc
import time
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sb
import xgboost as xgb
import lightgbm as lgb
import cloudpickle as cp
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchsummary import summary
from torch.utils.data import DataLoader, Dataset

from tqdm.auto import tqdm
from scipy import stats
from numerapi import NumerAPI
from scipy.stats import pearsonr
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.tree import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.decomposition import *
from sklearn.preprocessing import *
from sklearn.neural_network import *
from sklearn.model_selection import *
from sklearn.cluster._kmeans import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_xgboost():
    xgboost = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=5,
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.01,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=2,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    return xgboost

def get_random_forest():
    random_forest = RandomForestClassifier(
        n_estimators=1000,              
        max_depth=10,                   
        min_samples_split=5,            
        min_samples_leaf=4,            
        max_features='sqrt',           
        bootstrap=True,               
        class_weight='balanced',       
        n_jobs=-1,                     
        random_state=42,               
        max_samples=0.9,               
        oob_score=True,                
        criterion='gini',              
        verbose=1                      
    )
    return random_forest

def get_adaboost():
    base_estimator = DecisionTreeClassifier(
        class_weight='balanced',                
        max_features='sqrt',
        max_depth=3,    
        random_state=42         
    )

    adaboost = AdaBoostClassifier(
        estimator=base_estimator, 
        n_estimators=1000,             
        learning_rate=0.01,           
        algorithm='SAMME.R',           
        random_state=42               
    )
    return adaboost

def get_logistic_regression():
    logistic = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        solver='saga',
        penalty='l2',
        C=0.25,
        tol=1e-4,
        max_iter=500,
        n_jobs=-1
    )
    return logistic

def get_catboost():
    catboost = CatBoostClassifier(
        auto_class_weights='Balanced',
        random_state=42,
        iterations=500,
        learning_rate=0.1,
        depth=5,
        l2_leaf_reg=3,
        eval_metric='MultiClass',
        early_stopping_rounds=20,
        verbose=100
    )
    return catboost

def get_histogram_gb():
    histogram_gb = HistGradientBoostingClassifier(
        class_weight='balanced',
        random_state=42,
        learning_rate=0.001,
        max_iter=100,
        max_depth=10,
        min_samples_leaf=20,
        max_leaf_nodes=40,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        scoring='roc_auc_ovr',
        loss='log_loss',
        verbose=1
    )
    return histogram_gb

def get_lightgbm():
    lightGBM = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=10000,
        learning_rate=0.01,
        max_depth=10,
        num_leaves=2**10,
        colsample_bytree=0.1,
        random_state=42
    )
    return lightGBM

