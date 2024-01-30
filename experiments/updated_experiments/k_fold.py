import gensim
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from data_handling import preprocess, get_augmented_data, get_data

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

random_state = 42
np.random.seed(random_state)

def vectorize(X_train, X_test, vectorization, n_gram, max_df, min_df):
    if vectorization == 'bow':
        vectorizer = CountVectorizer(lowercase=True, ngram_range=[1,n_gram], max_df=max_df, min_df=min_df)         
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)            

    elif vectorization == 'tfidf':
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=[1,n_gram], max_df=max_df, min_df=min_df)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)              
    
    elif vectorization == 'word2vec':
        X_train = X_train.apply(lambda x: gensim.utils.simple_preprocess(x))
        X_test = X_test.apply(lambda x: gensim.utils.simple_preprocess(x))
                
        w2v_model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count = 2)
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
        
        X_train, X_test = X_train_vect_avg, X_test_vect_avg
    
    return X_train, X_test


def predict(X_train, Y_train, X_test, Y_test, model, random_state=42):
    if model == 'lr':
        lr = LogisticRegression(max_iter = 10000, class_weight = 'balanced', random_state=random_state)
        lr.fit(X_train, Y_train)
        Y_pred = lr.predict(X_test)        
    
    elif model == 'svm':
        clf = SVC(C=9, kernel='linear', degree=3, gamma='auto', class_weight='balanced', random_state=random_state)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)        
    
    elif model == 'mnb':
        mnb = MultinomialNB()
        mnb.fit(X_train, Y_train)
        Y_pred = mnb.predict(X_test)
    
    elif model == 'gnb':
        gnb = GaussianNB()
        gnb.fit(X_train, Y_train)
        Y_pred = gnb.predict(X_test)    
                
    elif model == 'lr+svm':
        lr = LogisticRegression(max_iter = 10000, random_state=random_state, class_weight='balanced')
        SVM = SVC(C=9, kernel='linear', degree=3, gamma='auto', class_weight='balanced', random_state=random_state)
            
        ensemble = VotingClassifier(estimators=[('lr', lr), ('svm', SVM)], voting='hard')
        ensemble.fit(X_train, Y_train)
            
        Y_pred = ensemble.predict(X_test)        

    elif model == 'lr+mnb':
        lr = LogisticRegression(max_iter = 10000, random_state=random_state, class_weight='balanced')
        mnb = MultinomialNB()
            
        ensemble = VotingClassifier(estimators=[('lr', lr), ('mnb', mnb)], voting='hard')
        ensemble.fit(X_train, Y_train)
            
        Y_pred = ensemble.predict(X_test)         
        
    elif model == 'lr+gnb':
        lr = LogisticRegression(max_iter = 10000, random_state=random_state, class_weight='balanced')
        gnb = GaussianNB()
            
        ensemble = VotingClassifier(estimators=[('lr', lr), ('mnb', gnb)], voting='hard')
        ensemble.fit(X_train, Y_train)
            
        Y_pred = ensemble.predict(X_test)          
    elif model == 'gnb+svm':
        SVM = SVC(C=9, kernel='linear', degree=3, gamma='auto', class_weight='balanced', random_state=random_state)
        gnb = GaussianNB()
            
        ensemble = VotingClassifier(estimators=[('svm', SVM), ('mnb', gnb)], voting='hard')
        ensemble.fit(X_train, Y_train)
            
        Y_pred = ensemble.predict(X_test)         
   
    elif model == 'mnb+svm':
        SVM = SVC(C=9, kernel='linear', degree=3, gamma='auto', class_weight='balanced', random_state=random_state)
        mnb = MultinomialNB()
            
        ensemble = VotingClassifier(estimators=[('svm', SVM), ('mnb', mnb)], voting='hard')
        ensemble.fit(X_train, Y_train)
            
        Y_pred = ensemble.predict(X_test)           

        
    return Y_pred


def cross_validate(k=5, augmented=False, vectorization='bow', model='lr', verbose=False, n_gram=1, max_df=0.9, min_df=2, random_state=42):
    accuracy = 0
    f1 = 0
    precision = 0
    recall = 0
    specificity = 0
    sensitivity = 0    
    
    for fold in range(1, k+1):
        if augmented:
            train_data, test_data = get_augmented_data(fold)
        else:
            train_data, test_data = get_data(fold)
        
        X_train, Y_train = train_data['text'], train_data['label']
        X_test, Y_test = test_data['text'], test_data['label']
        
        X_train, X_test = vectorize(X_train, X_test, vectorization, n_gram, max_df, min_df)
        Y_pred = predict(X_train, Y_train, X_test, Y_test, model, random_state=random_state)
        
        accuracy += accuracy_score(Y_test, Y_pred)/k
        f1 += f1_score(Y_test, Y_pred)/k
        precision += precision_score(Y_test, Y_pred)/k
        recall += recall_score(Y_test, Y_pred)/k
        
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
        specificity += (tn/(tn+fp))/k
        sensitivity += (tp/(tp+fn))/k
        
        if verbose:
            print(f"Fold {fold}")
            print(f"Accuracy -> {accuracy_score(Y_test, Y_pred):0.4f}")
            print(f"F1-Score -> {f1_score(Y_test, Y_pred):0.4f}")
            print(f"Precision -> {precision_score(Y_test, Y_pred):0.4f}")
            print(f"Recall -> {recall_score(Y_test, Y_pred):0.4f}")
            print(f"Specificity -> {tn/(tn+fp):0.4f}")
            print(f"Sensitivity -> {tp/(tp+fn):0.4f}")
    
    print('AVERAGED RESULTS')
    print(f"Accuracy -> {accuracy:0.4f}")
    print(f"F1-Score -> {f1:0.4f}")
    print(f"Precision -> {precision:0.4f}")
    print(f"Recall -> {recall:0.4f}")
    print(f"Specificity -> {specificity:0.4f}")
    print(f"Sensitivity -> {sensitivity:0.4f}")    
    return f1
        
        
            
            
        
            
        
            
    