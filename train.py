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

# In[4]:


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


# In[5]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# In[ ]:


from torchsummary import summary
from torch.utils.data import DataLoader, Dataset


# In[ ]:


from tqdm.auto import tqdm
from scipy import stats
from numerapi import NumerAPI
from scipy.stats import pearsonr
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# In[7]:


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

# In[11]:


feature_metadata = json.load(open(f"./data/v5.0/features.json"))

for metadata in feature_metadata:
  print(metadata, len(feature_metadata[metadata]))


# In[12]:


feature_sets = feature_metadata["feature_sets"]

for feature_set in ["small", "medium", "all"]:
  print(feature_set, len(feature_sets[feature_set]))


# In[13]:


feature_sets = feature_metadata["feature_sets"]
feature_sets.keys()


# In[14]:


for feature_set in feature_sets:
  print(f'Feature Set: {feature_set:<25}', f'Size: {len(feature_sets[feature_set])}')


# Loading the Training Set, with a 'medium' feature set

# In[15]:


feature_set = feature_sets["medium"]

train = pd.read_parquet(
    f"./data/v5.0/train.parquet",
    columns=["era", "target"] + feature_set
)


# Preprocessing the Training Set

# In[16]:


train.rename(columns=lambda x: f'feature {feature_set.index(x)}' if x in feature_set else x, inplace=True)
feature_set = train.columns.drop(["era", "target"])


# In[17]:


train['era'] = train['era'].astype('int32')


# In[18]:


train


# In[ ]:


train.isna().any().any()


# In[ ]:


train = train.dropna(subset=['target'])
train


# In[19]:


unique_era = train['era'].unique()


# In[20]:


train[train['era'] == unique_era[0]]


# In[21]:


dataset = train
dataset


# In[22]:


dataset.isna().any().any()


# In[23]:


dataset['target'].value_counts()


# Splitting the Training Set into Training & Validation Fractions

# In[24]:


train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

print(f'Train Split: {train_df.shape}')
print(f'Val Split: {val_df.shape}')


# Clearing the unused variables using Python's Garbage Collector to free up RAM

# In[25]:


del train
gc.collect()


# In[26]:


train_df['target'].value_counts()


# In[27]:


val_df['target'].value_counts()


# In[28]:


train_df


# In[29]:


val_df


# Encoding the Numeric Values in the Target into corresponding labels (class 0 to class 4)

# In[30]:


label_encoder = LabelEncoder()
label_encoder.fit(dataset['target'])
train_df['target'] = label_encoder.transform(train_df['target'])
val_df['target'] = label_encoder.transform(val_df['target'])


# Label Map/Dictionary to store the mapping

# In[31]:


labels_dict = {i: label_encoder.transform([i])[0] for i in label_encoder.classes_}
labels_dict


# In[32]:


train_df_x = train_df.drop(['target'], axis=1, inplace=False)
train_df_y = train_df['target']


# In[33]:


train_df_x


# In[34]:


train_df_y


# In[35]:


val_df_x = val_df.drop(['target'], axis=1, inplace=False)
val_df_y = val_df['target']


# In[36]:


val_df_x


# In[37]:


val_df_y


# In[38]:


del train_df
del val_df

gc.collect()


# Undersampling the training & validation fractions to ensure a more balanced dataset, since NumerAi is imbalanced/skewed towards a value of 0.5 (class 2)

# In[40]:


undersample = RandomUnderSampler(random_state=42)


# In[41]:


train_df_x_resampled, train_df_y_resampled = undersample.fit_resample(train_df_x, train_df_y)


# In[42]:


val_df_x_resampled, val_df_y_resampled = undersample.fit_resample(val_df_x, val_df_y)


# In[43]:


del train_df_x
del val_df_x
del train_df_y
del val_df_y

gc.collect()


# Converting data into floating type format

# In[44]:


train_df_x_resampled = train_df_x_resampled.astype(np.float32)
val_df_x_resampled = val_df_x_resampled.astype(np.float32)

train_df_y_resampled = train_df_y_resampled.astype(np.float32)
val_df_y_resampled = val_df_y_resampled.astype(np.float32)


# In[45]:


train_df_x_resampled


# In[46]:


val_df_x_resampled


# In[47]:


train_df_y_resampled


# In[48]:


val_df_y_resampled


# Function to extract the highly correlated features with respect to the predictions

# In[49]:


def get_highly_correlated_features(x: pd.DataFrame, y, correlation_threshold: float = 0.03) -> list:
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=x.index)

    correlations = x.apply(lambda feature: feature.corr(y))
    highly_correlated_features = correlations[correlations.abs() > correlation_threshold].index.tolist()
    
    return highly_correlated_features


# Function to neutralise the effect of highly correlated features with respect to the predictions by subtracting a proportion from the predictions

# In[50]:


def neutralize_predictions(predictions: np.ndarray, x: pd.DataFrame, features_to_neutralize: list, proportion: float = 1.0) -> np.ndarray:
	predictions_df = pd.DataFrame(predictions, columns=['prediction'], index=x.index)
	neutralized_preds = predictions_df['prediction'].copy()
	
	for feature in features_to_neutralize:
		neutralizer = x[feature]
		adjustment = proportion * (neutralizer.dot(predictions_df['prediction']) / neutralizer.dot(neutralizer)) * neutralizer
		neutralized_preds -= adjustment
	# neutralizer = x
	# print(neutralizer.shape, neutralized_preds.shape)
	# adjustment = proportion * (neutralizer.dot(predictions_df['prediction']) / neutralizer.dot(neutralizer)) * neutralizer
	# neutralized_preds -= adjustment
   
	return neutralized_preds.values


# Calculating the class weights to ensure a more balanced training for some of the weak classifiers/experts

# In[53]:


classes = np.unique(train_df_y_resampled)
class_weights = compute_class_weight('balanced', classes=classes, y=train_df_y_resampled)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
sample_weights = np.array([class_weight_dict[label] for label in train_df_y_resampled])


# Evaluation Functions

# Function to compute Label Frequencies

# In[ ]:


def label_frequency(predictions):
    unique, counts = np.unique(predictions, return_counts=True)
    label_frequencies = dict(zip(unique, counts))
    print("Label frequencies:", label_frequencies)


# Function to compute NumerAi Correlation

# In[ ]:


def numerai_corr(preds, target):
  ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
  gauss_ranked_preds = stats.norm.ppf(ranked_preds)

  centered_target = target - target.mean()

  preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
  target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5

  return np.corrcoef(preds_p15, target_p15)[0, 1]


# EXPERT-1 (XGBOOST CLASSIFIER)

# In[54]:


'''
Commented out if the model is already trained over this & saved
'''

expert1 = get_xgboost()

expert1.fit(train_df_x_resampled, train_df_y_resampled, sample_weight=sample_weights)


# Loading the Model

# In[55]:


# with open('./saved_models/numerai_expert1.pkl', 'rb') as f:
#     expert1 = pickle.load(f)
# print("Model loaded successfully!")


# Saving the Model

# In[56]:


with open('./saved_models/numerai_expert1.pkl', 'wb') as f:
    pickle.dump(expert1, f)
print("Expert1 saved successfully!")


# Generating Predictions from Expert-1

# In[57]:


expert1_pred = expert1.predict(val_df_x_resampled)


# In[58]:


expert1_pred


# Computing Label Frequencies in Prediction, Accuracy, Highly Correlated Features & Neutralising the Predictions

# In[59]:


label_frequency(expert1_pred)


# In[60]:


acc = accuracy_score(expert1_pred, val_df_y_resampled)
print("Accuracy on Training Set: ", acc)


# In[61]:


highly_correlated_features = get_highly_correlated_features(val_df_x_resampled, expert1_pred, correlation_threshold=0.09)
print("Highly correlated features:", highly_correlated_features)


# In[62]:


# expert1_pred = neutralize_predictions(predictions=expert1_pred, x=val_df_x_resampled, features_to_neutralize=highly_correlated_features, proportion=0.01)
# print("Neutralized predictions, Expert1:", expert1_pred)


# In[ ]:


expert1_pred = neutralize_predictions(predictions=expert1_pred, x=val_df_x_resampled, features_to_neutralize=feature_set, proportion=0.0001)
print("Neutralized predictions, Expert1:", expert1_pred)


# EXPERT-2 (RANDOM FOREST CLASSIFIER)

# In[63]:


'''
Commented out if the model is already trained over this & saved
'''

expert2 = get_random_forest()

expert2.fit(train_df_x_resampled, train_df_y_resampled, sample_weight=sample_weights)


# Loading the Model

# In[64]:


# with open('./saved_models/numerai_expert2.pkl', 'rb') as f:
#     expert2 = pickle.load(f)
# print("Model loaded successfully!")


# Saving the Model

# In[65]:


with open('./saved_models/numerai_expert2.pkl', 'wb') as f:
    pickle.dump(expert2, f)
print("Expert2 saved successfully!")


# Generating Predictions from Expert-2

# In[66]:


expert2_pred = expert2.predict(val_df_x_resampled)


# In[67]:


expert2_pred


# Computing Label Frequencies in Prediction, Accuracy, Highly Correlated Features & Neutralising the Predictions

# In[68]:


label_frequency(expert2_pred)


# In[69]:


acc = accuracy_score(expert2_pred, val_df_y_resampled)
print("Accuracy on Training Set: ", acc)


# In[70]:


highly_correlated_features = get_highly_correlated_features(val_df_x_resampled, expert2_pred, correlation_threshold=0.09)
print("Highly correlated features:", highly_correlated_features)


# In[71]:


# expert2_pred = neutralize_predictions(predictions=expert2_pred, x=val_df_x_resampled, features_to_neutralize=highly_correlated_features, proportion=0.01)
# print("Neutralized predictions, Expert2:", expert2_pred)


# In[ ]:


expert2_pred = neutralize_predictions(predictions=expert2_pred, x=val_df_x_resampled, features_to_neutralize=feature_set, proportion=0.0001)
print("Neutralized predictions, Expert2:", expert2_pred)


# EXPERT-3 (ADABOOST CLASSIFIER with DECISION TREE CLASSIFIER as BASE ESTIMATOR)

# In[72]:


'''
Commented out if the model is already trained over this & saved
'''

expert3 = get_adaboost()

expert3.fit(train_df_x_resampled, train_df_y_resampled)


# Loading the Model

# In[73]:


# with open('./saved_models/numerai_expert3.pkl', 'rb') as f:
#     expert3 = pickle.load(f)
# print("Model loaded successfully!")


# Saving the Model

# In[74]:


with open('./saved_models/numerai_expert3.pkl', 'wb') as f:
    pickle.dump(expert3, f)
print("Expert3 saved successfully!")


# Generating Predictions from Expert-3

# In[75]:


expert3_pred = expert3.predict(val_df_x_resampled)


# In[76]:


expert3_pred


# Computing Label Frequencies in Prediction, Accuracy, Highly Correlated Features & Neutralising the Predictions

# In[77]:


label_frequency(expert3_pred)


# In[78]:


acc = accuracy_score(expert3_pred, val_df_y_resampled)
print("Accuracy on Training Set: ", acc)


# In[79]:


highly_correlated_features = get_highly_correlated_features(val_df_x_resampled, expert3_pred, correlation_threshold=0.09)
print("Highly correlated features:", highly_correlated_features)


# In[80]:


# expert3_pred = neutralize_predictions(predictions=expert3_pred, x=val_df_x_resampled, features_to_neutralize=highly_correlated_features, proportion=0.01)
# print("Neutralized predictions, Expert3:", expert3_pred)


# In[ ]:


expert3_pred = neutralize_predictions(predictions=expert3_pred, x=val_df_x_resampled, features_to_neutralize=feature_set, proportion=0.0001)
print("Neutralized predictions, Expert3:", expert3_pred)


# EXPERT-4 (LOGISTIC REGRESSION)

# In[81]:


'''
Commented out if the model is already trained over this & saved
'''

expert4 = get_logistic_regression()

expert4.fit(train_df_x_resampled, train_df_y_resampled)


# Loading the Model

# In[82]:


# with open('./saved_models/numerai_expert4.pkl', 'rb') as f:
#     expert4 = pickle.load(f)
# print("Model loaded successfully!")


# Saving the Model

# In[83]:


with open('./saved_models/numerai_expert4.pkl', 'wb') as f:
    pickle.dump(expert4, f)
print("Expert4 saved successfully!")


# Generating Predictions from Expert-4

# In[84]:


expert4_pred = expert4.predict(val_df_x_resampled)


# In[85]:


expert4_pred


# Computing Label Frequencies in Prediction, Accuracy, Highly Correlated Features & Neutralising the Predictions

# In[86]:


label_frequency(expert4_pred)


# In[87]:


acc = accuracy_score(expert4_pred, val_df_y_resampled)
print("Accuracy on Training Set: ", acc)


# In[88]:


highly_correlated_features = get_highly_correlated_features(val_df_x_resampled, expert4_pred, correlation_threshold=0.09)
print("Highly correlated features:", highly_correlated_features)


# In[89]:


# expert4_pred = neutralize_predictions(predictions=expert4_pred, x=val_df_x_resampled, features_to_neutralize=highly_correlated_features, proportion=0.01)
# print("Neutralized predictions, Expert4:", expert4_pred)


# In[ ]:


expert4_pred = neutralize_predictions(predictions=expert4_pred, x=val_df_x_resampled, features_to_neutralize=feature_set, proportion=0.0001)
print("Neutralized predictions, Expert4:", expert4_pred)


# EXPERT-5 (CATBOOST CLASSIFIER)

# In[91]:


'''
Commented out if the model is already trained over this & saved
'''

expert5 = get_catboost()

expert5.fit(train_df_x_resampled, train_df_y_resampled)


# Loading the Model

# In[92]:


# with open('./saved_models/numerai_expert5.pkl', 'rb') as f:
#     expert5 = pickle.load(f)
# print("Model loaded successfully!")


# Saving the Model

# In[93]:


with open('./saved_models/numerai_expert5.pkl', 'wb') as f:
    pickle.dump(expert5, f)
print("Expert5 saved successfully!")


# Generating Predictions from Expert-5

# In[94]:


expert5_pred = expert5.predict(val_df_x_resampled)


# In[95]:


if expert5_pred.ndim > 1:
    expert5_pred = expert5_pred.ravel()


# In[96]:


expert5_pred


# Computing Label Frequencies in Prediction, Accuracy, Highly Correlated Features & Neutralising the Predictions

# In[97]:


label_frequency(expert5_pred)


# In[98]:


acc = accuracy_score(expert5_pred, val_df_y_resampled)
print("Accuracy on Training Set: ", acc)


# In[99]:


highly_correlated_features = get_highly_correlated_features(val_df_x_resampled, expert5_pred, correlation_threshold=0.09)
print("Highly correlated features:", highly_correlated_features)


# In[100]:


# expert5_pred = neutralize_predictions(predictions=expert5_pred, x=val_df_x_resampled, features_to_neutralize=highly_correlated_features, proportion=0.01)
# print("Neutralized predictions, Expert5:", expert5_pred)


# In[ ]:


expert5_pred = neutralize_predictions(predictions=expert5_pred, x=val_df_x_resampled, features_to_neutralize=feature_set, proportion=0.0001)
print("Neutralized predictions, Expert5:", expert5_pred)


# EXPERT-6 (HISTOGRAM-BASED GRADIENT BOOST CLASSIFIER)

# In[101]:


'''
Commented out if the model is already trained over this & saved
'''

expert6 = get_histogram_gb()

expert6.fit(train_df_x_resampled, train_df_y_resampled, sample_weight=sample_weights)


# Loading the Model

# In[102]:


# with open('./saved_models/numerai_expert6.pkl', 'rb') as f:
#     expert6 = pickle.load(f)
# print("Model loaded successfully!")


# Saving the Model

# In[103]:


with open('./saved_models/numerai_expert6.pkl', 'wb') as f:
    pickle.dump(expert6, f)
print("Expert6 saved successfully!")


# Generating Predictions from Expert-6

# In[104]:


expert6_pred = expert6.predict(val_df_x_resampled)


# In[105]:


expert6_pred


# Computing Label Frequencies in Prediction, Accuracy, Highly Correlated Features & Neutralising the Predictions

# In[106]:


label_frequency(expert6_pred)


# In[107]:


acc = accuracy_score(expert6_pred, val_df_y_resampled)
print("Accuracy on Training Set: ", acc)


# In[108]:


highly_correlated_features = get_highly_correlated_features(val_df_x_resampled, expert6_pred, correlation_threshold=0.09)
print("Highly correlated features:", highly_correlated_features)


# In[109]:


# expert6_pred = neutralize_predictions(predictions=expert6_pred, x=val_df_x_resampled, features_to_neutralize=highly_correlated_features, proportion=0.01)
# print("Neutralized predictions, Expert6:", expert6_pred)


# In[ ]:


expert6_pred = neutralize_predictions(predictions=expert6_pred, x=val_df_x_resampled, features_to_neutralize=feature_set, proportion=0.0001)
print("Neutralized predictions, Expert6:", expert6_pred)


# META-MODEL (LIGHTGBM)

# In[110]:


meta_model = get_lightgbm()


# In[ ]:


meta_val_x = np.column_stack((expert1_pred, expert2_pred, expert3_pred, expert4_pred, expert5_pred, expert6_pred))


# In[111]:


meta_val_x


# In[112]:


meta_model.fit(meta_val_x, val_df_y_resampled)


# Loading the Meta-Model

# In[ ]:


# with open('./saved_models/numerai_meta_model.pkl', 'rb') as f:
#     meta_model = pickle.load(f)
# print("Meta Model loaded successfully!")


# Saving the Meta-Model

# In[113]:


with open('./saved_models/numerai_meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)
print("Meta Model saved successfully!")


# Testing Performance on Trained Data

# In[114]:


meta_y_pred = meta_model.predict(meta_val_x)


# In[115]:


meta_y_pred


# In[116]:


bins = [0.5, 1.5, 2.5, 3.5]

rounded_predictions = np.digitize(meta_y_pred, bins)


# In[117]:


rounded_predictions


# Computing Relevant Evaluation Metrics

# In[118]:


label_frequency(rounded_predictions)


# In[119]:


acc = accuracy_score(rounded_predictions, val_df_y_resampled)
print("Accuracy on Training Set: ", acc)


# Pearson's Correlation

# In[120]:


pearson_corr, _ = stats.pearsonr(rounded_predictions, val_df_y_resampled)
print("Pearson Correlation:", pearson_corr)


# Computing the NumerAi's Correlation Metric

# In[ ]:


rounded_predictions = pd.Series(rounded_predictions)


# In[122]:


actual_corr = numerai_corr(rounded_predictions, val_df_y_resampled)
actual_corr

