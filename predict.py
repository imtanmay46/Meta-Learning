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

# In[ ]:


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


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[ ]:


from torchsummary import summary
from torch.utils.data import DataLoader, Dataset


# In[ ]:


from scipy import stats
from tqdm.auto import tqdm
from datetime import datetime
from numerapi import NumerAPI
from scipy.stats import pearsonr
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# In[ ]:


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


# Function to compute NumerAi Correlation

# In[ ]:


def label_frequency(predictions):
    unique, counts = np.unique(predictions, return_counts=True)
    label_frequencies = dict(zip(unique, counts))
    print("Label frequencies:", label_frequencies)


# Creating a Feature Set

# In[11]:


feature_metadata = json.load(open(f"./data/v5.0/features.json"))

for metadata in feature_metadata:
  print(metadata, len(feature_metadata[metadata]))


# In[ ]:


feature_sets = feature_metadata["feature_sets"]
for feature_set in ["small", "medium", "all"]:
  print(feature_set, len(feature_sets[feature_set]))


# In[ ]:


feature_sets = feature_metadata["feature_sets"]
feature_sets.keys()


# In[ ]:


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


# Loading the Live Data Set, with a 'medium' feature set

# In[24]:


feature_set = feature_sets["medium"]

live_data = pd.read_parquet(f"./data/v5.0/live.parquet")
features = [f for f in live_data.columns if "feature" in f]
live = live_data[["era"] + feature_set]


# Preprocessing the Live Set (in the same manner as the Training & Validation Sets)

# In[25]:


live.rename(columns=lambda x: f'feature {feature_set.index(x)}' if x in feature_set else x, inplace=True)


# In[26]:


live


# Setting a Numeric Value to the 'era' column

# In[27]:


live['era'] = 1


# In[28]:


live


# Generating Predictions from Experts

# EXPERT-1 (XGBOOST CLASSIFIER)

# In[29]:


expert1_live_pred = expert1.predict(live)


# In[30]:


expert1_live_pred


# EXPERT-2 (RANDOM FOREST CLASSIFIER)

# In[31]:


expert2_live_pred = expert2.predict(live)


# In[32]:


expert2_live_pred


# EXPERT-3 (ADABOOST CLASSIFIER with DECISION TREE CLASSIFIER as BASE ESTIMATOR)

# In[33]:


expert3_live_pred = expert3.predict(live)


# In[34]:


expert3_live_pred


# EXPERT-4 (LOGISTIC REGRESSION)

# In[35]:


expert4_live_pred = expert4.predict(live)


# In[36]:


expert4_live_pred


# EXPERT-5 (CATBOOST CLASSIFIER)

# In[37]:


expert5_live_pred = expert5.predict(live)


# In[ ]:


if expert5_live_pred.ndim > 1:
    expert5_live_pred = expert5_live_pred.ravel()


# In[ ]:


expert5_live_pred


# EXPERT-6 (HISTOGRAM-BASED GRADIENT BOOST CLASSIFIER)

# In[39]:


expert6_live_pred = expert6.predict(live)


# In[40]:


expert6_live_pred


# Generating Predictions from Meta-Model (LIGHTGBM)

# In[41]:


meta_live_x = np.column_stack((expert1_live_pred, expert2_live_pred, expert3_live_pred, expert4_live_pred, expert5_live_pred, expert6_live_pred))


# In[42]:


meta_live_x


# In[43]:


meta_live_predictions = meta_model.predict(meta_live_x)


# In[44]:


meta_live_predictions


# In[45]:


bins = [0.5, 1.5, 2.5, 3.5]

live_predictions = np.digitize(meta_live_predictions, bins)


# In[46]:


live_predictions


# In[47]:


label_frequency(live_predictions)


# Converting the Classes back into Numeric Form for Submission to NumerAi Platform

# Creating a Label Map (Both forward & reverse Mapping)

# In[48]:


labels_dict = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}
reverse_labels_dict = {v: k for k, v in labels_dict.items()}
og_live_predictions = np.vectorize(reverse_labels_dict.get)(live_predictions)


# In[49]:


og_live_predictions


# In[50]:


label_frequency(og_live_predictions)


# Preparing Data into Submission Format

# In[51]:


submission = pd.Series(og_live_predictions, index=live.index).to_frame("prediction")

submission


# Saving the Live Predictions into Pickle & CSV formats

# In[52]:


predictions_dir = "./predictions/"
os.makedirs(predictions_dir, exist_ok=True)

current_date = datetime.now().strftime("%d-%m-%Y")
prediction_file_path = os.path.join(predictions_dir, f"{current_date}_predictions.csv")

# with open(prediction_file_path, "wb") as f:
#     f.write(cloudpickle.dumps(submission))

# print(f"Predictions saved to '{prediction_file_path}' using cloudpickle.")


# In[53]:


submission = submission.reset_index()
submission = submission.rename(columns={'index': 'id'})
submission.to_csv(prediction_file_path, index=False)

print(f"Predictions saved to '{prediction_file_path}'")

