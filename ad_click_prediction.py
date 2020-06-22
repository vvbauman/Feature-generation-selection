# -*- coding: utf-8 -*-
"""
CODE FOR AD CLICK PREDICTION JUPYTER NOTEBOOK
Intended for anyone who doesn't want the supplementary text within the notebook

Dataset is from Avazu Click-Through Rate Prediction contest on Kaggle (must download the data on your machine to run this code)
Use dataset to predict whether someone will click on an ad (binary classification)

Topics covered/emphasized are count encoding of categorical features, generation of interaction features, and feature selection
Feature selection methods covered are L1 regularization, univariate selection, and decision trees for feature selection

Jun 16, 2020
V. Bauman
"""

import pandas as pd
import numpy as np

import category_encoders as ce
import itertools

from sklearn.model_selection import train_test_split #stratifies the split
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, balanced_accuracy_score #balanced_accuracy_score gives average recall for both classes. dataset is imbalanced

def train_eval_tree(x_train, y_train, x_devel, y_devel):
    """
    Train and evaluate performance (training and development/test sets) of decision tree model.
    
    Parameters
    ----------
    x_train : pd.DataFrame where the rows are the examples and the columns are features for the training set
    y_train : pd.Series with corresponding labels for x_train
    x_devel : pd.DataFrame where the rows are the examples and the columns are features for the devleopment set
    y_devel : pd.Series with corresponding labels for x_devel

    Returns
    -------
    None.
    (Prints to screen precision and recall on training and development sets)

    """
    tree= DecisionTreeClassifier(random_state= 0, class_weight= 'balanced')
    tree.fit(x_train, y_train)
    train_predicts= tree.predict(x_train)
    train_recall= balanced_accuracy_score(y_train, train_predicts)
    train_precision= precision_score(y_train, train_predicts)
    
    devel_predicts= tree.predict(x_devel)
    devel_recall= balanced_accuracy_score(y_devel, devel_predicts)
    devel_precision= precision_score(y_devel, devel_predicts)
    
    print('Training data precision:', train_precision, 'Training data recall:', train_recall)
    print('Development data precision:', devel_precision, 'Development data recall:', devel_recall)
    return

#load data
#change path to wherever you have the data file saved
file= pd.read_csv('C:\\Users\\16479\\Documents\\ad_clicks\\train_small_version.csv', delimiter=',')
data= file.drop(['click'], axis= 1) #feature values only
labels= file['click']
N,d= np.shape(data)

#see datatype of each of the features. any categorical features will be count encoded
#print(data.dtypes)

#for every column, see how many unique entries there are 
#if the number of unique entries is >25% of the number of examples, that column won't be included as part of the feature extraction
#for categories that have less than 25 unique values, create interaction features 
pop_cols= np.array([]) #store the names of columns that won't be used in ML model
pop_thresh= N/4
interact_feats= np.array([]) #store names of columns that will be used to create interaction features
for i in data.columns:
    unique_col_entries= data[i].nunique()
    #print(i, unique_col_entries)
    if unique_col_entries > pop_thresh:
        pop_cols= np.append(pop_cols, i)
    if unique_col_entries < 25:
        interact_feats= np.append(interact_feats, i)

#FEATURE GENERATION - COUNT ENCODE CATEGORICAL FEATURES
#for all columns that are categorical (entries are strings), get count encoding
categ_feats= data.select_dtypes(include= ['category', object]).columns
count_enc= ce.CountEncoder(cols= categ_feats)
count_enc.fit(data[categ_feats])

#use only numerical features in ML model
pop_cols= np.append(pop_cols, categ_feats)

#FEATURE GENERATION - CREATE INTERACTION FEATURES
#convert any numerical columns that meet this criterion to string then create new columns of strings that represent interactions and count encode them
interact_df= pd.DataFrame(index= data.index)
for col1, col2 in itertools.combinations(interact_feats,2):
    interact_col= "_".join([col1, col2])
    interact_vals= data[col1].map(str)+ '_'+ data[col2].map(str)
    interact_df[interact_col]= ce.CountEncoder().fit_transform(interact_vals)

#add the categorical data that was encoded earlier to the end of the dataframe:
data= data.join(count_enc.transform(data[categ_feats]).add_suffix("_count"))
#add the interaction terms to the end of the dataframe
data= data.join(interact_df)
#drop all columns in pop_cols
data.drop(pop_cols, axis=1, inplace= True)

#FEATURE SELECTION
#first split data into training and validation sets (use 80/20 split)
x_train, x_devel, y_train, y_devel= train_test_split(data, labels, test_size= 0.2, random_state= 0)
N,d= np.shape(x_train)

#METHOD 0: no feature selection; all features are used in the ML model
print('No feature selection:')
train_eval_tree(x_train, y_train, x_devel, y_devel)

#METHOD 1: LASSO/L1 regularization
lsvc= LinearSVC(C= 1.0, penalty= 'l1', dual= False, max_iter= 1000).fit(x_train, y_train)
svc_mod= SelectFromModel(lsvc, prefit= True)
x_train_new= svc_mod.transform(x_train)

#get the selected/most important features and extract from validation set
selected_feats= pd.DataFrame(svc_mod.inverse_transform(x_train_new), index= x_train.index, columns= x_train.columns)
selected_cols= selected_feats.columns[selected_feats.var() != 0]
x_devel_new= x_devel[selected_cols]

#now train and test a decision tree using these selected features
print('L1 regularization:')
train_eval_tree(x_train_new, y_train, x_devel_new, y_devel)

#METHOD 2: SelectKBest using the f_classif score
select_feats= SelectKBest(f_classif, k= 10)
x_train_new= select_feats.fit_transform(x_train, y_train)
selected_feats= pd.DataFrame(select_feats.inverse_transform(x_train_new), index= x_train.index, columns= x_train.columns)
selected_cols= selected_feats.columns[selected_feats.var() != 0]
x_devel_new= x_devel[selected_cols]
print('Univariate feature selection (f_classif):')
train_eval_tree(x_train_new, y_train, x_devel_new, y_devel)

#METHOD 3: RANDOM FOREST
forest= RandomForestClassifier(n_estimators= 1000, random_state= 0)
forest.fit(x_train, y_train)
selector= SelectFromModel(forest, threshold= 0.10)
selector.fit(x_train, y_train)
for important_feats in selector.get_support(indices= True):
    print(x_train.columns[important_feats])
x_train_new= selector.transform(x_train)
x_devel_new= selector.transform(x_devel)

print('Random forest feature selection:')
train_eval_tree(x_train_new, y_train, x_devel_new, y_devel)

