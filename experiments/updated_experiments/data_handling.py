import gensim
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def preprocess(texts):
    pass


def get_augmented_data(fold):
    train_data_path = f"k_fold_data/fold_{fold}_augmented_train.csv"
    test_data_path = f"k_fold_data/fold_{fold}_test.csv"
    train_data = pd.read_csv(train_data_path, header=0, names=["study_id", "text", "label"], encoding='unicode_escape')
    test_data = pd.read_csv(test_data_path, header=0, names=["study_id", "text", "label"], encoding='unicode_escape')
    
    return train_data, test_data



def get_data(fold):
    train_data_path = f"k_fold_data/fold_{fold}_train.csv"
    test_data_path = f"k_fold_data/fold_{fold}_test.csv"
    train_data = pd.read_csv(train_data_path, header=0, names=["study_id", "text", "label"], encoding='unicode_escape')
    test_data = pd.read_csv(test_data_path, header=0, names=["study_id", "text", "label"], encoding='unicode_escape')
    
    return train_data, test_data
    