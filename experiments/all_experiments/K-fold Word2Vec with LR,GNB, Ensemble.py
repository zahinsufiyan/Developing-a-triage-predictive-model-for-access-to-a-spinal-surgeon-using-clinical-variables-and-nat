#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[1]:


import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 100)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# In[2]:


random_state = 100
np.random.seed(random_state)


# In[6]:


# Loading dataset
df2 = pd.read_csv(r"KECSACQIProject_DATA_2022-03-31_full dataset for CS.csv",header=0, names=["study_id", "Outcome", "mr_report"],encoding='unicode_escape')
df2['mr_report'] = df2['mr_report'].astype(str)

df2.head()


# In[7]:


ax = sns.countplot(x="Outcome", data=df2)


# In[8]:


# Cleaning data using the built in cleaner in gensim
df2['text_clean'] = df2['mr_report'].apply(lambda x: gensim.utils.simple_preprocess(x))
df2.head()


# # Logistic Regression: Five-fold cross-validation

# In[9]:


k = 5
fold = 0

accuracy = 0
f1 = 0
precision = 0
recall = 0

folds = KFold(n_splits=k, random_state=None, shuffle=False)

for train_index, test_index in folds.split(df2['mr_report'], df2['Outcome']):
    fold += 1
    X_train, Y_train = df2['mr_report'][train_index], df2['Outcome'][train_index]
    X_test, Y_test = df2['mr_report'][test_index], df2['Outcome'][test_index]
    
    
    # Training the word2vec model
    w2v_model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count=2)
    
    # Generate aggregated sentence vectors based on the word vectors for each word in the sentence
    words = set(w2v_model.wv.index_to_key )
    X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train])
    X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test])
    
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
        
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    
    
    # fit the training dataset on the LR
    lr = LogisticRegression(max_iter = 10000, class_weight = 'balanced', random_state=0)
    lr.fit(X_train_vect_avg, Y_train.values.ravel())

    # predicting the labels on validation dataset
    Y_pred = lr.predict(X_test_vect_avg)
        
    accuracy += accuracy_score(Y_test, Y_pred)/k
    f1 += f1_score(Y_test, Y_pred)/k
    precision += precision_score(Y_test, Y_pred)/k
    recall += recall_score(Y_test, Y_pred)/k
    
    #Specificity and Sensitivity
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    
    
    print(f"Fold {fold}")
    print(f"Accuracy -> {accuracy_score(Y_test, Y_pred):0.4f}")
    print(f"F1-Score -> {f1_score(Y_test, Y_pred):0.4f}")
    print(f"Precision -> {precision_score(Y_test, Y_pred):0.4f}")
    print(f"Recall -> {recall_score(Y_test, Y_pred):0.4f}")
    
    print(f"Specificity -> {specificity:0.4f}")
    print(f"Sensitivity -> {sensitivity:0.4f}")


# In[10]:


print(f"Accuracy -> {accuracy:0.4f}")
print(f"F1-Score -> {f1:0.4f}")
print(f"Precision -> {precision:0.4f}")
print(f"Recall -> {recall:0.4f}")


# # GNB: Five-fold cross-validation

# In[28]:


k = 5
fold = 0

accuracy = 0
f1 = 0
precision = 0
recall = 0

folds = KFold(n_splits=k, random_state=None, shuffle=False)

for train_index, test_index in folds.split(df2['mr_report'], df2['Outcome']):
    fold += 1
    X_train, Y_train = df2['mr_report'][train_index], df2['Outcome'][train_index]
    X_test, Y_test = df2['mr_report'][test_index], df2['Outcome'][test_index]
    
    
    # Training the word2vec model
    w2v_model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count=2)
    
    # Generate aggregated sentence vectors based on the word vectors for each word in the sentence
    words = set(w2v_model.wv.index_to_key )
    X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train])
    X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test])
    
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
        
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    
    
    # fit the training dataset on the LR
    gnb = GaussianNB()
    gnb.fit(X_train_vect_avg, Y_train.values.ravel())

    # predicting the labels on validation dataset
    Y_pred = gnb.predict(X_test_vect_avg)
        
    accuracy += accuracy_score(Y_test, Y_pred)/k
    f1 += f1_score(Y_test, Y_pred)/k
    precision += precision_score(Y_test, Y_pred)/k
    recall += recall_score(Y_test, Y_pred)/k
    
    #Specificity and Sensitivity
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    
    
    print(f"Fold {fold}")
    print(f"Accuracy -> {accuracy_score(Y_test, Y_pred):0.4f}")
    print(f"F1-Score -> {f1_score(Y_test, Y_pred):0.4f}")
    print(f"Precision -> {precision_score(Y_test, Y_pred):0.4f}")
    print(f"Recall -> {recall_score(Y_test, Y_pred):0.4f}")
    
    print(f"Specificity -> {specificity:0.4f}")
    print(f"Sensitivity -> {sensitivity:0.4f}")


# In[29]:


print(f"Accuracy -> {accuracy:0.4f}")
print(f"F1-Score -> {f1:0.4f}")
print(f"Precision -> {precision:0.4f}")
print(f"Recall -> {recall:0.4f}")


# # LR + SVM Ensemble: Five-fold cross-validation

# In[30]:


k = 5
fold = 0

accuracy = 0
f1 = 0
precision = 0
recall = 0

folds = KFold(n_splits=k, random_state=None, shuffle=False)

for train_index, test_index in folds.split(df2['mr_report'], df2['Outcome']):
    fold += 1
    X_train, Y_train = df2['mr_report'][train_index], df2['Outcome'][train_index]
    X_test, Y_test = df2['mr_report'][test_index], df2['Outcome'][test_index]
    
    
    # Training the word2vec model
    w2v_model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count=2)
    
    # Generate aggregated sentence vectors based on the word vectors for each word in the sentence
    words = set(w2v_model.wv.index_to_key )
    X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train])
    X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test])
    
    X_train_vect_avg = []
    for v in X_train_vect:
        if v.size:
            X_train_vect_avg.append(v.mean(axis=0))
        else:
            X_train_vect_avg.append(np.zeros(100, dtype=float))
        
    X_test_vect_avg = []
    for v in X_test_vect:
        if v.size:
            X_test_vect_avg.append(v.mean(axis=0))
        else:
            X_test_vect_avg.append(np.zeros(100, dtype=float))
    
    clf1 = LogisticRegression(multi_class='multinomial', class_weight = 'balanced', random_state=0)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    
    # fit the training dataset on the LR
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf3)], voting='hard')
    eclf1.fit(X_train_vect_avg, Y_train.values.ravel())

    # predicting the labels on validation dataset
    Y_pred = eclf1.predict(X_test_vect_avg)
        
    accuracy += accuracy_score(Y_test, Y_pred)/k
    f1 += f1_score(Y_test, Y_pred)/k
    precision += precision_score(Y_test, Y_pred)/k
    recall += recall_score(Y_test, Y_pred)/k
    
    #Specificity and Sensitivity
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    
    
    print(f"Fold {fold}")
    print(f"Accuracy -> {accuracy_score(Y_test, Y_pred):0.4f}")
    print(f"F1-Score -> {f1_score(Y_test, Y_pred):0.4f}")
    print(f"Precision -> {precision_score(Y_test, Y_pred):0.4f}")
    print(f"Recall -> {recall_score(Y_test, Y_pred):0.4f}")
    
    print(f"Specificity -> {specificity:0.4f}")
    print(f"Sensitivity -> {sensitivity:0.4f}")


# In[31]:


print(f"Accuracy -> {accuracy:0.4f}")
print(f"F1-Score -> {f1:0.4f}")
print(f"Precision -> {precision:0.4f}")
print(f"Recall -> {recall:0.4f}")


# In[ ]:




