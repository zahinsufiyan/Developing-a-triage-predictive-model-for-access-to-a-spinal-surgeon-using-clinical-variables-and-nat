#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score, recall_score


# In[2]:


import nltk
nltk.download('punkt')


# In[3]:


import nltk
nltk.download('wordnet')


# In[4]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[5]:


import nltk
nltk.download('stopwords')


# In[12]:


np.random.seed(200)


# In[13]:


df = pd.read_csv(r"D:\E\Baikka\Project\KEC SAC radiology data for CS 8.3.2022.csv",encoding='utf-8')
df['mr_report'] = df['mr_report'].astype(str)


# In[14]:


df.head()


# In[15]:


# dropiing few negative outcomes to maintain balance
remove_n = 136
drop_indices = np.random.choice(df[(df.Outcome == 0)].index, remove_n, replace=False)
print(drop_indices)
df2 = df.drop(drop_indices)
df2.reset_index(inplace=True)
df2.shape


# In[16]:


df2[:300]


# # Data pre-processing

# In[17]:


# Removing blank rows if any.
df2['mr_report'].dropna(inplace=True)


# In[18]:


# Changing all the text to lower case.
df2['mr_report'] = [entry.lower() for entry in df2['mr_report']]


# In[19]:


# Tokenization : Each entry in the mri_report will be broken into set of words
df2['mr_report']= [word_tokenize(entry) for entry in df2['mr_report']]


# In[20]:


# Removing Stop words, Non-Numeric and perfom Word Stemming/Lemmenting

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV



for index,entry in enumerate(df2['mr_report']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    df2.loc[index,'text_final'] = str(Final_words)


# In[21]:


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df2['text_final'],df2['Outcome'],test_size=0.2)


# In[22]:


print(Train_X[:10])


# In[23]:


print(Train_Y[:10])


# In[24]:


# Term Frequency: This summarizes how often a given word appears within a document
# Inverse Document Frequency: This down scales words that appear a lot across documents.

# fit the TG-IDF model on the whole mri_report. 
# This will help TF-IDF build a vocabulary of words which it has learned from the MRI data 
# and it will assign a unique integer number to each of these words.


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df2['text_final'])

# Finally we will transform Train_X and Test_X to vectorized Train_X_Tfidf and Test_X_Tfidf
# Each row will be a list of unique integer number and its associated importance calculated by TF-IDF.
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[25]:


#  the vocabulary that it has learned from the mri_report
print(Tfidf_vect.vocabulary_)


# In[26]:


# Printing the vectorized data
print(Train_X_Tfidf)


# # Using Support Vector Machine
# 

# In[30]:


# fit the training dataset on the SVM
SVM = svm.SVC(C=9, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

# predicting the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print("F1-Score -> ",f1_score(predictions_SVM, Test_Y))


# In[31]:


# Precision and Recall

precision = precision_score(Test_Y, predictions_SVM)
recall = recall_score(Test_Y, predictions_SVM)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((predictions_SVM==Test_Y).sum()/len(predictions_SVM), 3)))


# In[32]:


c_range = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

c_scores = []
f_scores = []

# we will loop through reasonable values of c
for c in c_range:
    
    # fit the training dataset on the SVM
    SVM = svm.SVC(C=c, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    # predicting the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    
    accuracy = accuracy_score(predictions_SVM, Test_Y)*100
    f1 = f1_score(predictions_SVM, Test_Y)
    
    c_scores.append(accuracy)
    f_scores.append(f1)
    
    print("SVM Accuracy Score and F1-Score when c=", c, "->", accuracy_score(predictions_SVM, Test_Y)*100, "and", f1_score(predictions_SVM, Test_Y))


# # Using Logistic Regression

# In[35]:


# fit the training dataset on the LR
lr = LogisticRegressionCV(cv=5, class_weight = 'balanced', random_state=0)
lr.fit(Train_X_Tfidf,Train_Y)

# predicting the labels on validation dataset
predictions_LR = lr.predict(Test_X_Tfidf)

print("LR Accuracy Score -> ",accuracy_score(predictions_LR, Test_Y)*100)
print("F1-Score -> ",f1_score(predictions_LR, Test_Y))


# In[37]:


# Precision and Recall

precision = precision_score(Test_Y, predictions_LR)
recall = recall_score(Test_Y, predictions_LR)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((predictions_LR==Test_Y).sum()/len(predictions_LR), 3)))


# # Using Voting Ensembles

# In[38]:


clf1 = LogisticRegression(multi_class='multinomial', class_weight = 'balanced', random_state=0)
clf2 = svm.SVC(C=9, kernel='linear', degree=3, gamma='auto')

eclf1 = VotingClassifier(estimators=[('lr', clf1), ('svm', clf2)], voting='hard')
eclf1 = eclf1.fit(Train_X_Tfidf,Train_Y)
predictions_eclf1 = eclf1.predict(Test_X_Tfidf)
print("Hard voting Accuracy Score -> ",accuracy_score(predictions_eclf1, Test_Y)*100)
print("Hard voting F1-Score -> ",f1_score(predictions_eclf1, Test_Y))

precision = precision_score(Test_Y, predictions_eclf1)
recall = recall_score(Test_Y, predictions_eclf1)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((predictions_eclf1==Test_Y).sum()/len(predictions_eclf1), 3)))


# In[ ]:




