#!/usr/bin/env python
# coding: utf-8

# Meta Learning
# 
# NumerAi
# 
# Tanmay Singh
# 2021569
# CSAI
# Class of '25

# In[1]:


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


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[4]:


from torchsummary import summary
from torch.utils.data import DataLoader, Dataset


# In[5]:


from tqdm.auto import tqdm
from scipy import stats
from numerapi import NumerAPI
from scipy.stats import pearsonr
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# In[6]:


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


# Creating a Feature Set

# In[10]:


feature_metadata = json.load(open(f"./data/v5.0/features.json"))

for metadata in feature_metadata:
  print(metadata, len(feature_metadata[metadata]))


# In[11]:


feature_sets = feature_metadata["feature_sets"]
for feature_set in ["small", "medium", "all"]:
  print(feature_set, len(feature_sets[feature_set]))


# In[12]:


feature_sets = feature_metadata["feature_sets"]
feature_sets.keys()


# In[13]:


for feature_set in feature_sets:
  print(f'Feature Set: {feature_set:<25}', f'Size: {len(feature_sets[feature_set])}')


# Loading the Saved Experts & the Meta-Model

# In[14]:


with open('./saved_models/numerai_expert1.pkl', 'rb') as f:
    expert1 = pickle.load(f)
print("Model loaded successfully!")


# In[15]:


with open('./saved_models/numerai_expert2.pkl', 'rb') as f:
    expert2 = pickle.load(f)
print("Model loaded successfully!")


# In[16]:


with open('./saved_models/numerai_expert3.pkl', 'rb') as f:
    expert3 = pickle.load(f)
print("Model loaded successfully!")


# In[17]:


with open('./saved_models/numerai_expert4.pkl', 'rb') as f:
    expert4 = pickle.load(f)
print("Model loaded successfully!")


# In[18]:


with open('./saved_models/numerai_expert5.pkl', 'rb') as f:
    expert5 = pickle.load(f)
print("Model loaded successfully!")


# In[19]:


with open('./saved_models/numerai_expert6.pkl', 'rb') as f:
    expert6 = pickle.load(f)
print("Model loaded successfully!")


# In[20]:


with open('./saved_models/numerai_meta_model.pkl', 'rb') as f:
    meta_model = pickle.load(f)
print("Meta Model loaded successfully!")


# Loading the Validation Set, with a 'medium' feature set

# In[22]:


feature_set = feature_sets["medium"]

val = pd.read_parquet(
    f"./data/v5.0/validation.parquet",
    columns=["era", "target"] + feature_set
)


# Preprocessing the Validation Set (in the same manner as the Training Set)

# In[23]:


val.rename(columns=lambda x: f'feature {feature_set.index(x)}' if x in feature_set else x, inplace=True)
feature_set = val.columns.drop(["era", "target"])


# In[24]:


val['era'] = val['era'].astype('int32')


# In[25]:


val


# In[26]:


val.isna().any().any()


# In[27]:


val = val.dropna(subset=['target'])
val


# In[28]:


unique_era = val['era'].unique()


# In[ ]:


val[val['era'] == unique_era[0]]


# In[29]:


test_set = val
test_set


# In[30]:


test_set.isna().any().any()
test_set['target'].value_counts()


# Encoding the Numeric Values in the Target into corresponding labels (class 0 to class 4)

# In[31]:


label_encoder = LabelEncoder()
label_encoder.fit(test_set['target'])
test_set['target'] = label_encoder.transform(test_set['target'])


# In[32]:


test_df_x = test_set.drop(['target'], axis=1, inplace=False)
test_df_y = test_set['target']


# In[33]:


test_df_x


# In[34]:


test_df_y


# In[35]:


test_df_x_resampled = test_df_x

test_df_x_resampled


# In[36]:


test_df_y_resampled = test_df_y

test_df_y_resampled


# Function to compute Label Frequencies

# In[37]:


def label_frequency(predictions):
    unique, counts = np.unique(predictions, return_counts=True)
    label_frequencies = dict(zip(unique, counts))
    print("Label frequencies:", label_frequencies)


# Function to compute NumerAi Correlation

# In[38]:


def numerai_corr(preds, target):
  ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
  gauss_ranked_preds = stats.norm.ppf(ranked_preds)

  centered_target = target - target.mean()

  preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
  target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5

  return np.corrcoef(preds_p15, target_p15)[0, 1]


# Generating Predictions from Experts

# EXPERT-1 (XGBOOST CLASSIFIER)

# In[44]:


expert1_test_pred = expert1.predict(test_df_x_resampled)


# In[45]:


expert1_test_pred


# EXPERT-2 (RANDOM FOREST CLASSIFIER)

# In[46]:


expert2_test_pred = expert2.predict(test_df_x_resampled)


# In[47]:


expert2_test_pred


# EXPERT-3 (ADABOOST CLASSIFIER with DECISION TREE CLASSIFIER as BASE ESTIMATOR)

# In[48]:


expert3_test_pred = expert3.predict(test_df_x_resampled)


# In[49]:


expert3_test_pred


# EXPERT-4 (LOGISTIC REGRESSION)

# In[50]:


expert4_test_pred = expert4.predict(test_df_x_resampled)


# In[51]:


expert4_test_pred


# EXPERT-5 (CATBOOST CLASSIFIER)

# In[52]:


expert5_test_pred = expert5.predict(test_df_x_resampled)


# In[53]:


if expert5_test_pred.ndim > 1:
    expert5_test_pred = expert5_test_pred.ravel()


# In[54]:


expert5_test_pred


# EXPERT-6 (HISTOGRAM-BASED GRADIENT BOOST CLASSIFIER)

# In[55]:


expert6_test_pred = expert6.predict(test_df_x_resampled)


# In[56]:


expert6_test_pred


# Generating Predictions from the Meta-Model (LIGHTGBM)

# In[57]:


meta_test_x = np.column_stack((expert1_test_pred, expert2_test_pred, expert3_test_pred, expert4_test_pred, expert5_test_pred, expert6_test_pred))


# In[58]:


meta_test_x


# In[59]:


meta_test_y_pred = meta_model.predict(meta_test_x)


# In[60]:


meta_test_y_pred


# In[61]:


bins = [0.5, 1.5, 2.5, 3.5]

rounded_predictions = np.digitize(meta_test_y_pred, bins)


# In[62]:


rounded_predictions


# In[63]:


label_frequency(rounded_predictions)


# In[ ]:


test_df_y_resampled


# Computing Relevant Evaluation Metrics

# In[64]:


acc = accuracy_score(rounded_predictions, test_df_y_resampled)
print("Accuracy on Validation Set: ", acc)


# Pearson's Correlation

# In[65]:


pearson_corr, _ = stats.pearsonr(rounded_predictions, test_df_y_resampled)
print("Pearson Correlation:", pearson_corr)


# Reporting Class-wise Accuracies & F1 Scores

# In[66]:


class_accuracies = {}

for class_label in np.unique(test_df_y_resampled):
    class_mask = (test_df_y_resampled == class_label)
    class_accuracy = accuracy_score(test_df_y_resampled[class_mask], rounded_predictions[class_mask])
    class_accuracies[class_label] = class_accuracy
    print(f"Accuracy for class {class_label}: {class_accuracy:.4f}")

print("\n")

f1_scores = f1_score(test_df_y_resampled, rounded_predictions, average=None)
for class_label, f1 in zip(np.unique(test_df_y_resampled), f1_scores):
    print(f"F1 Score for class {class_label}: {f1:.4f}")


# Saving the Predictions in a Pickle File

# In[ ]:


# with open('numerai_fullprediction.pkl', 'wb') as f:
#     pickle.dump(rounded_predictions, f)

# print("Predictions saved successfully to numerai_fullprediction.pkl!")


# Computing the NumerAi's Correlation Metric

# In[67]:


rounded_predictions = pd.Series(rounded_predictions)


# In[68]:


actual_corr = numerai_corr(rounded_predictions, test_df_y_resampled)
actual_corr

