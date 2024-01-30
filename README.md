# Spine-Assessment-Project
The Spine Assessment Clinic (SAC) in Edmonton, AB determines whether a patient experiencing
chronic back or neck pain could benefit from spine surgery. This decision is based on a
family doctor’s description of the problem, a patient questionnaire, MRI results, and discussion
with the patient. The aim of this paper is to support the decision making process of the SAC by
developing a machine learning system that could identify those patients who need surgery based
on the textual MRI reports. We preprocess the texts before vectorization using bag-of-words,
term frequency–inverse document frequency, or word2vec techniques. The generated vectors
are used as the features for machine learning algorithms including logistic regression, SVM, an
ensemble of the above two, and complement na¨ıve Bayes. Our best performing system used
word2vec text vectorization and logistic regression. We report a sensitivity of 0.55 and specificity
of 0.58 for this system using five-fold cross validation. Our work provides a foundation for
future research on this task. Possible avenues include refinement of the text preprocesing step
and further investigation of data augmentation techniques to address the class imbalance of the
dataset.

# Paper Link: https://link.springer.com/article/10.1007/s00586-023-07552-4
