#!/usr/bin/env python
# coding: utf-8

# In[22]:




# In[38]:


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


# In[24]:


# Loading dataset
df = pd.read_csv(r"KEC_SAC_radiology_data_for_CS_8.3.2022.csv",names=["study_id", "Outcome", "mr_report"],encoding='utf-8')
df['mr_report'] = df['mr_report'].astype(str)

df.head()


# In[25]:


# dropiing few negative outcomes to maintain balance
#remove_n = 136
#drop_indices = np.random.choice(df[(df.Outcome == 0)].index, remove_n, replace=False)
#print(drop_indices)
#df2 = df.drop(drop_indices)
#df2.reset_index(inplace=True)
#df2.shape
df2 = df


# In[26]:


# Cleaning data using the built in cleaner in gensim
df2['text_clean'] = df2['mr_report'].apply(lambda x: gensim.utils.simple_preprocess(x))
df2.head()


# In[27]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split (df2['text_clean'], df2['Outcome'] , test_size=0.2)


# In[28]:


# Training the word2vec model
w2v_model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count=2)


# In[29]:


# Representing all of the words that the Word2Vec model learned a vector for.
w2v_model.wv.index_to_key


# In[30]:


w2v_model.wv.most_similar('spine')


# In[31]:


# Generate aggregated sentence vectors based on the word vectors for each word in the sentence

words = set(w2v_model.wv.index_to_key )
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_train])
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words]) for ls in X_test])


# In[32]:


for i, v in enumerate(X_train_vect):
    print(len(X_train.iloc[i]), len(v))


# In[33]:


# Computing sentence vectors by averaging the word vectors for the words contained in the sentence

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


# In[34]:


for i, v in enumerate(X_train_vect_avg):
    print(len(X_train.iloc[i]), len(v))


# In[47]:


# fit the training dataset on the LR
lr = LogisticRegressionCV(cv=5, max_iter = 10000, class_weight = 'balanced', random_state=0)
lr.fit(X_train_vect_avg, y_train.values.ravel())

# predicting the labels on validation dataset
predictions_LR = lr.predict(X_test_vect_avg)

print("LR Accuracy Score -> ",accuracy_score(predictions_LR, y_test)*100)
print("F1-Score -> ",f1_score(predictions_LR, y_test))


# In[36]:


# Precision and Recall

precision = precision_score(y_test, predictions_LR)
recall = recall_score(y_test, predictions_LR)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((predictions_LR==y_test).sum()/len(predictions_LR), 3)))


# In[45]:


# fit the training dataset on the GNB
gnb = GaussianNB()
gnb.fit(X_train_vect_avg, y_train.values.ravel())

# predicting the labels on validation dataset
predictions_gnb = gnb.predict(X_test_vect_avg)

print("GNB Accuracy Score -> ",accuracy_score(predictions_gnb, y_test)*100)
print("F1-Score -> ",f1_score(predictions_gnb, y_test))


# In[46]:


# Precision and Recall

precision = precision_score(y_test, predictions_gnb)
recall = recall_score(y_test, predictions_gnb)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((predictions_gnb==y_test).sum()/len(predictions_gnb), 3)))


# # Using Voting Ensembles

# In[49]:


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




