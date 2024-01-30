import k_fold

vectorizations = ['bow', 'tfidf', 'word2vec']

models = {'bow': ['lr', 'svm', 'mnb', 'lr+svm', 'lr+mnb', 'mnb+svm'],
          'tfidf': ['lr', 'svm', 'mnb', 'lr+svm', 'lr+mnb', 'mnb+svm'],
          'word2vec': ['lr', 'svm', 'gnb', 'lr+svm', 'lr+gnb', 'gnb+svm']}

n_grams = {'bow': [1, 2],
           'tfidf': [1, 2],
           'word2vec': [1]}
best_f1 = 0
for vectorization in vectorizations:
    for model in models[vectorization]:
        for n_gram in n_grams[vectorization]:
            for augmented in [False, True]:
                print(vectorization, model, n_gram, augmented)
                f1 = k_fold.cross_validate(augmented=augmented, vectorization=vectorization, model=model, verbose=False, n_gram=n_gram, max_df=0.9, min_df=2, random_state=42)
                print()
                if f1 > best_f1:
                    best_f1 = f1
                    best_settings = [vectorization, model, n_gram, augmented]

print(best_settings)