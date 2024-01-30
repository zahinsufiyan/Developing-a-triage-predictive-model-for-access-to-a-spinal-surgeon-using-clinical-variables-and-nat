#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -U gensim


# In[2]:


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


# In[13]:


random_state = 100
np.random.seed(random_state)


# In[14]:


# Loading dataset
df = pd.read_csv(r"D:\E\Baikka\Project\New data\KEC SAC radiology data for CS 8.3.2022.csv",encoding='unicode_escape')
df['mr_report'] = df['mr_report'].astype(str)

df.head()


# In[15]:


ax = sns.countplot(x="Outcome", data=df)


# In[16]:


# dropiing few negative outcomes to maintain balance
remove_n = 227
drop_indices = np.random.choice(df[(df.Outcome == 0)].index, remove_n, replace=False)
print(drop_indices)
df2 = df.drop(drop_indices)
df2.reset_index(inplace=True)
df2.shape


# In[17]:


# Cleaning data using the built in cleaner in gensim
df2['text_clean'] = df2['mr_report'].apply(lambda x: gensim.utils.simple_preprocess(x))
df2.head()


# In[22]:


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
    lr = LogisticRegressionCV(cv=5, max_iter = 10000, class_weight = 'balanced', random_state=0)
    lr.fit(X_train_vect_avg, Y_train.values.ravel())

    # predicting the labels on validation dataset
    Y_pred = lr.predict(X_test_vect_avg)
        
    accuracy += accuracy_score(Y_pred, Y_test)/k
    f1 += f1_score(Y_pred, Y_test)/k
    precision += precision_score(Y_pred, Y_test)/k
    recall += recall_score(Y_pred, Y_test)/k
    
    print(f"Fold {fold}")
    print(f"Accuracy -> {accuracy_score(Y_pred, Y_test):0.4f}")
    print(f"F1-Score -> {f1_score(Y_pred, Y_test):0.4f}")
    print(f"Precision -> {precision_score(Y_pred, Y_test):0.4f}")
    print(f"Recall -> {recall_score(Y_pred, Y_test):0.4f}")


# In[23]:


print(f"Accuracy -> {accuracy:0.4f}")
print(f"F1-Score -> {f1:0.4f}")
print(f"Precision -> {precision:0.4f}")
print(f"Recall -> {recall:0.4f}")


# In[24]:


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
        
    accuracy += accuracy_score(Y_pred, Y_test)/k
    f1 += f1_score(Y_pred, Y_test)/k
    precision += precision_score(Y_pred, Y_test)/k
    recall += recall_score(Y_pred, Y_test)/k
    
    print(f"Fold {fold}")
    print(f"Accuracy -> {accuracy_score(Y_pred, Y_test):0.4f}")
    print(f"F1-Score -> {f1_score(Y_pred, Y_test):0.4f}")
    print(f"Precision -> {precision_score(Y_pred, Y_test):0.4f}")
    print(f"Recall -> {recall_score(Y_pred, Y_test):0.4f}")


# In[25]:


print(f"Accuracy -> {accuracy:0.4f}")
print(f"F1-Score -> {f1:0.4f}")
print(f"Precision -> {precision:0.4f}")
print(f"Recall -> {recall:0.4f}")


# In[27]:


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
        
    accuracy += accuracy_score(Y_pred, Y_test)/k
    f1 += f1_score(Y_pred, Y_test)/k
    precision += precision_score(Y_pred, Y_test)/k
    recall += recall_score(Y_pred, Y_test)/k
    
    print(f"Fold {fold}")
    print(f"Accuracy -> {accuracy_score(Y_pred, Y_test):0.4f}")
    print(f"F1-Score -> {f1_score(Y_pred, Y_test):0.4f}")
    print(f"Precision -> {precision_score(Y_pred, Y_test):0.4f}")
    print(f"Recall -> {recall_score(Y_pred, Y_test):0.4f}")


# In[28]:


print(f"Accuracy -> {accuracy:0.4f}")
print(f"F1-Score -> {f1:0.4f}")
print(f"Precision -> {precision:0.4f}")
print(f"Recall -> {recall:0.4f}")


# # Using Voting Ensembles

# In[19]:


from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression(multi_class='multinomial', class_weight = 'balanced', random_state=0)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(X_train_vect_avg, y_train.values.ravel())
predictions_eclf1 = eclf1.predict(X_test_vect_avg)
print("Hard voting Accuracy Score -> ",accuracy_score(predictions_eclf1, y_test)*100)
print("Hard voting F1-Score -> ",f1_score(predictions_eclf1, y_test))
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((predictions_eclf1==y_test).sum()/len(predictions_eclf1), 3)))


# In[ ]:




