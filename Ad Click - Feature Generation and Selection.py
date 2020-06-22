#!/usr/bin/env python
# coding: utf-8

# # Preamble
# This notebook illustrates how to use feature generation and feature selection to improve predictions from a machine learning model. The dataset used in this notebook is from the [Avazu Click-Through Rate Prediction contest](https://www.kaggle.com/c/avazu-ctr-prediction/overview) and is a binary classification dataset. The topics covered here are count encoding of categorical features, creation of interaction features, and three different approaches to feature selection. After each feature selection method, a decision tree is trained and its performance is compared to the performance of a decision tree trained using all available features. Data leakage isn't discussed here but the feature selection methods use the training data only.
# 
# # Imports and Loading the Dataset

# In[7]:


import pandas as pd
import numpy as np
import category_encoders as ce
import itertools
from sklearn.model_selection import train_test_split

file= pd.read_csv('C:\\Users\\16479\\Documents\\ad_clicks\\train.csv', delimiter= ',')
data= file.drop(['click'], axis= 1) #labels column
labels= file['click']
N,d= np.shape(data)
print(N,d)
print(labels.value_counts())
data.head()


# This dataset has **1,048,575 examples**, each characterized by **23 features**. The label for each example is one of **0** or **1** with there being many more instances of examples with the label 0.
# 
# It may be obvious by looking at the head of the dataset, but let's see what data types we're working with for each of the features. Also, since scikit learn machine learning models can't handle NaNs, let's ensure that there are no NaNs as any of the feature values.

# In[8]:


print(data.dtypes)
if data.isnull().values.any() == False:
    print('There are no NaNs in the dataset')
else:
    print('There are NaNs in the dataset. Must address before proceeding since most machine learning models cannot handle NaN values')


# Those features that are of type "object" are categorical while those that are of either type "float64" and "int64" are numerical. Since not all machine learning models can handle categorical features, we will encode these categorical features later on.

# # Feature Elimination
# Now that we have a nice overview of the dataset we're working with, let's see if there are any features that we want to exclude and if there are any features that we should make interaction features with. For every feature, we'll count how many unique entries there are. If the number of unique entries is greater than 25% of the number of examples, that particular feature will be excluded. Having ~250,000 (25% x 1,048,575) unique entries for one feature means that, for each possible value of this feature, only ~4 examples have the same value, meaning the feature likely isn't very informative for what we're trying to predict. Just like how having all examples having the same feature value for a particular feature isn't very useful for making predictions, having all examples having different values for the same feature likely isn't useful either.
# 
# Also for every feature, if the number of unique values is less than 25, we'll create interaction features (later but we'll get the names of those particular columns now).

# In[11]:


pop_cols= np.array([]) #store names of columns that won't be used in ML model
pop_thresh= N/4
interact_feats= np.array([]) #store names of cols to make interaction features with later
for i in data.columns:
    unique_col_entries= data[i].nunique()
    if unique_col_entries > pop_thresh:
        pop_cols= np.append(pop_cols, i)
    elif unique_col_entries < 25:
        interact_feats= np.append(interact_feats, i)
        
print('Features to eliminate:', pop_cols)
print('Features to create interaction features with:', interact_feats)


# # Feature Generation
# ## Count Encode Categorical Features
# For all categorical features, all feature values will be **count-encoded**. This means that, for each categorical feature, an integer of the total number of times a particular level appears in the dataset will replace every instance of that level.

# In[12]:


categ_feats= data.select_dtypes(include= ["category", object]).columns
count_enc= ce.CountEncoder(cols= categ_feats)
count_enc.fit(data[categ_feats])


# Now that we've expressed the categorical features in a quantitative way, we won't use the original categorical variables in our machine learning model.

# In[13]:


pop_cols= np.append(pop_cols, categ_feats)


# ## Create Interaction Features
# For all original features that have <25 unique entries, we'll create interaction features among them. First we'll convert any numerical features that meet this criterion to string, then we'll create new columns of strings that represent interactions, and then we'll count encode these so our final interaction feature values are numerical.

# In[14]:


#dataframe to hold interaction features. 
#need index to match that of the original feature values in order to combine later
interact_df= pd.DataFrame(index= data.index)
for feat1, feat2 in itertools.combinations(interact_feats, 2):
    interact_col= "_".join([feat1, feat2])
    interact_vals= data[feat1].map(str) + "_" + data[feat2].map(str)
    interact_df[interact_col]= ce.CountEncoder().fit_transform(interact_vals)


# Now that we've generated these count encoded features, let's add them to the end of our original dataframe that contains all of the original feature values. We'll also drop all of the features/columns we don't want to include to make our predictions in our machine learning model.

# In[15]:


#count encoded categorical features
data= data.join(count_enc.transform(data[categ_feats]).add_suffix("count"))
#count encoded interaction features
data= data.join(interact_df)
#drop columns
data.drop(pop_cols, axis= 1, inplace= True)
print(np.shape(data))


# # Feature Selection
# The feature generation process resulted in us doubling the number of features! There's a possibility that not all of these features will be helpful to us trying to complete this binary classification task. We will try three different approaches to feature selection and see what our prediction performance is like with each one. For each approach, the machine learning model will be a decision tree with the same hyperparameter settings. Since the dataset is imbalanced, the performance metrics will be precision and recall.
# 
# First let's split the dataset into a training and validation set (80/20 split) (the Avazu dataset has a separate test set file that hasn't been included in this notebook but should be used as the test set once you're happy with your feature selection approach and trained machine learning model). We'll also train our decision tree using all features to have a baseline to compare the three approaches to.

# In[18]:


x_train, x_devel, y_train, y_devel= train_test_split(data, labels, test_size= 0.2, random_state= 0)
N,d= np.shape(x_train)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, balanced_accuracy_score

def train_eval_tree(x_train, y_train, x_devel, y_devel):
    """Function that trains and evaluates the performance of a decision tree
    Inputs: training data feature values, training data labels, development data feature values, development data labels
    Outputs: None (model performance on training and development sets printed to screen)
    """
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

tree= DecisionTreeClassifier(random_state= 0, class_weight= "balanced")
train_eval_tree(x_train, y_train, x_devel, y_devel)


# ## Method 1: L1 Regularization
# This feature selection method involves training a linear model that uses an L1 penalty. All features are used to train this model and the L1 penalty causes the weight/contribution of unimportant features to be zero. We then extract the non-zeroed features and use them in our decision tree. An important note on this feature selection method is that it considers all features and how they collectively contribute to each prediction.

# In[19]:


#train linear model with L1 penalty
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

lsvc= LinearSVC(C= 1.0, penalty= 'l1', dual= False).fit(x_train, y_train)
svc_mod= SelectFromModel(lsvc, prefit= True)

#get training set that contains only the non-zeroed features
x_train_new= svc_mod.transform(x_train)

#get development set that contains only the non-zeroed features
selected_feats= pd.DataFrame(svc_mod.inverse_transform(x_train_new), index= x_train.index, columns= x_train.columns)
selected_cols= selected_feats.columns[selected_feats.var() != 0]
x_devel_new= x_devel[selected_cols]

#out of curiousity, see which features were retained
print('Features retained:',selected_cols)
print('Number of features retained:',np.shape(selected_cols)[0])

#train and test decision tree using only these features
tree.fit(x_train_new, y_train)
train_eval_tree(x_train_new, y_train, x_devel_new, y_devel)


# Compared to our baseline, our performance improved a bit! We can see that we're not overfitting to the training data and that our recall improved. Pretty cool considering we didn't make any changes to the model itself, just the features we were providing to it. Let's see if we see the same thing with the other two methods!
# 
# ## Method 2: SelectKBest using the f_classif score
# This feature selection method involves evaluating the linear relationship between each feature and the label/target. The top-k features with the strongest relationship with the label are identified and are used in our decision tree. We will use the top 10 best features since this was the number of features retained when the L1 regularization method was used. Unlike the L1 regularization method, this is a univariate method, meaning that each feature is considered individually/one-by-one.

# In[20]:


#get the 10 best features
from sklearn.feature_selection import SelectKBest, f_classif
select_feats= SelectKBest(f_classif, k= 10)

#get training set that has only the top 10 features
x_train_new= select_feats.fit_transform(x_train, y_train)

#get development set that has only the top 10 features
selected_feats= pd.DataFrame(select_feats.inverse_transform(x_train_new), index= x_train.index, columns= x_train.columns)
selected_cols= selected_feats.columns[selected_feats.var() != 0]
x_devel_new= x_devel[selected_cols]

#out of curiousity, see which features were retained
print('Features retained:',selected_cols)

#train and test decision tree using only these features
tree.fit(x_train_new, y_train)
train_eval_tree(x_train_new, y_train, x_devel_new, y_devel)


# Here we also see a bit on an improvement in model performance compared to our baseline. We also see that the feature set from the two feature selection methods is different. For the L1 regularization method, only features from the original 23 features were identified as being the best for this particular prediction task. For the SelectKBest-f_classif method, features from the original feature set as well as some of the features from our feature generation methods were among the top 10 most important features. For example, "C1_app_category" is the interaction between the features "C1" and "app_category" and "app_idcount" is the count encoded version of the original feature "app_id".
# 
# ## Method 3: Decision Trees/Random Forests for Feature Selection
# The final feature selection method considered in this notebook is the decision tree. This method is favourable because decision trees naturally rank features by how well they distinguish classes. Features that best split/distinguish classes are evaluated at nodes at the base/start of a tree. Therefore, if we prune a tree at a certain node, we can get a subset of the most informative features.
# 
# Implementing this feature selection method involves training a decision tree or random forest (an ensemble of decision trees) and identifying the features that have an importance greater than some threshold. These features are the ones used in your machine learning model (in this case, also a decision tree!). For this example, an arbitrary threshold of 0.10 will be used.

# In[21]:


#train a random forest using all features and get the most important features
from sklearn.ensemble import RandomForestClassifier

forest= RandomForestClassifier(n_estimators= 1000, random_state= 0)
forest.fit(x_train, y_train)
selector= SelectFromModel(forest, threshold= 0.10)
selector.fit(x_train, y_train)

#out of curiousity, see which features were retained and their importance
for important_feats in selector.get_support(indices= True):
    print(x_train.columns[important_feats])
x_train_new= selector.transform(x_train)
x_devel_new= selector.transform(x_devel)
    
#train and test decision tree using only these features
tree.fit(x_train_new, y_train)
train_eval_tree(x_train_new, y_train, x_devel_new, y_devel)


# # Conclusions
# This notebook presented some feature generation and feature selection techniques. This notebook is not exhaustive but it demonstrates that identifying important features via feature selection can improve the performance of a machine learning model without having to make any changes to the model itself. There's no one-size-fits-all approach to feature generation and feature selection, but hopefully this provides a start to improving your models' performance.
