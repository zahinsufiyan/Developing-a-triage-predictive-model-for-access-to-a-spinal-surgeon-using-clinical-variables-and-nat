from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

random_state = 42
np.random.seed(random_state)

tune = False
unigram = True
bigram = False
trigram = False


def split_data(data):
    # split the data into 80% train, 20% validation
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=random_state)
    X_train, t_train = train_data["text"], train_data["label"]
    X_val, t_val = val_data["text"], val_data["label"]
    return X_train, t_train, X_val, t_val


def tune_logistic_regression(X_train, t_train, X_val, t_val):
    """
    Tune some hyperparameters for logistic regression, incuding penalty, solver, and C (regularization parameter).
    Picks the combination that maximizes f1 score.
    :param X_train: training data inputs
    :param t_train: training data outputs
    :param X_val: validation data inputs
    :param t_val: validation data ouptuts
    :return: a pd.DataFrame of the results for each tuning combination.
    """
    results = pd.DataFrame(columns=["penalty", "solver", "C", "f1", "precision", "recall", "accuracy"])

    penalties = ['l1', 'l2', 'none']

    solvers = {'l1': ['liblinear'],
               'l2': ['newton-cg', 'liblinear'],
               'none': ['newton-cg']}

    C_values = {'l1': [x / 100 for x in range(5, 101, 1)],
                'l2': [x / 100 for x in range(5, 101, 1)],
                'none': [1]}  # 'none' ignores C because there is no regularization
    print(C_values)
    best_f1 = -1
    for penalty in penalties:
        for solver in solvers[penalty]:
            for C in C_values[penalty]:
                # train and validate logisitc regression
                clf = LogisticRegression(penalty=penalty, solver=solver, C=C, random_state=random_state, class_weight='balanced').fit(X_train,
                                                                                                             t_train)
                t_hat = clf.predict(X_val)

                # calculate metrics
                precision, recall, f1, support = precision_recall_fscore_support(y_true=t_val, y_pred=t_hat,
                                                                                 average="binary")
                accuracy = accuracy_score(y_true=t_val, y_pred=t_hat)

                pd.concat([results,
                           pd.DataFrame([penalty, solver, C, f1, precision, recall, accuracy])], ignore_index=True)

                # store current best settings and results
                if f1 > best_f1:
                    best_f1, best_precision, best_recall, best_accuracy = f1, precision, recall, accuracy
                    best_penalty, best_solver, best_C = penalty, solver, C

    print("Logistic Regression Tuning")
    print("-" * 60)
    print(f"Best penalty: {best_penalty}")
    print(f"Best solver: {best_solver}")
    print(f"Best C: {best_C}")
    display_results(best_f1, best_precision, best_recall, best_accuracy)

    return results


def display_results(f1, precision, recall, accuracy):
    print("-" * 60)
    print(f"F1 score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("-" * 60 + "\n")


def balance_data(data):
    indices = np.random.choice(data[(data.label == 0)].index, 100, replace=False)
    data = data.drop(indices)
    data.reset_index(inplace=True)
    return data


def k_fold_cross_validation(data):
    skf = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)
    fold = 0
    f1s = []
    precisions = []
    recalls = []
    accuracies = []
    results = {}

    for train_index, test_index in skf.split(data['text'], data['label']):
        fold += 1
        X_train, t_train = data['text'][train_index], data['label'][train_index]
        X_test, t_test = data['text'][test_index], data['label'][test_index]
        if unigram:
            vectorizer = CountVectorizer(max_df=0.9, min_df=0.02)
            penalty = 'l1'
            solver = 'liblinear'
            C = 0.16

        if bigram:
            vectorizer = CountVectorizer(ngram_range=[1, 2], max_df=0.9, min_df=0.02)
            penalty = 'l1'
            solver = 'liblinear'
            C = 0.18
        if trigram:
            vectorizer = CountVectorizer(ngram_range=[1, 3], max_df=0.9, min_df=0.02)
            penalty = 'l1'
            solver = 'liblinear'
            C = 0.17

        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        clf = LogisticRegression(penalty=penalty, solver=solver, C=C, random_state=random_state, class_weight='balanced').fit(X_train,
                                                                                                     t_train)
        t_hat = clf.predict(X_test)

        # calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true=t_test, y_pred=t_hat,
                                                                         average="binary")
        accuracy = accuracy_score(y_true=t_test, y_pred=t_hat)

        results[fold] = {'f1': f1,
                         'precision': precision,
                         'recall': recall,
                         'accuracy': accuracy}
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)

    print(f"Average F1: {np.mean(f1s):.4f}")
    print(f"Average precision: {np.mean(precisions):.4f}")
    print(f"Average recall: {np.mean(recalls):.4f}")
    print(f"Average accuracy: {np.mean(accuracies):.4f}")
    return results


def main():
    data_path = Path(str(Path("KEC_SAC_radiology_data_for_CS_8.3.2022.csv")))
    data = pd.read_csv(data_path, header=0, names=["study_id", "label", "text"])

    if unigram:
        vectorizer = CountVectorizer(max_df=0.9, min_df=0.02)
    if bigram:
        vectorizer = CountVectorizer(ngram_range=[1, 2], max_df=0.85, min_df=0.01)
    if trigram:
        vectorizer = CountVectorizer(ngram_range=[1, 3], max_df=0.85, min_df=0.01)

    if tune:
        X_train, t_train, X_val, t_val = split_data(data)
        X_train = vectorizer.fit_transform(X_train)
        X_val = vectorizer.transform(X_val)
        vocabulary = vectorizer.get_feature_names_out()
        print(vocabulary)
        tune_logistic_regression(X_train, t_train, X_val, t_val)

    else:
        results = k_fold_cross_validation(data)


if __name__ == "__main__":
    main()
