# Spine Assessment Project

## Overview
Welcome to the Spine Assessment Project. This project aims to enhance the decision-making process at the Spine Assessment Clinic (SAC) in Edmonton, AB. SAC is instrumental in determining the necessity of spine surgery for patients with chronic back or neck pain.

## Project Objective
The goal is to develop a machine learning system that can identify patients requiring surgery based on textual MRI reports.

## Methodology
### Data Preprocessing
- We preprocess MRI report texts using techniques such as bag-of-words, term frequency-inverse document frequency (TF-IDF), and word2vec.

### Machine Learning Models
- **Implemented Algorithms**: Logistic regression, Support Vector Machine (SVM), an ensemble of both, and complement naïve Bayes.
- **Feature Engineering**: Utilizes preprocessed text vectors as input features.

### Best Performing System
- Our most effective model combines word2vec text vectorization with logistic regression.
- **Performance Metrics**: Achieved a sensitivity of 0.55 and specificity of 0.58, verified through five-fold cross-validation.

## Conclusion and Future Work
- This project serves as a foundation for further research in this field.
- Future directions include refining text preprocessing and exploring data augmentation to tackle class imbalance in the dataset.

## Additional Information
- **Research Paper**: For more in-depth details, [read our published paper](https://link.springer.com/article/10.1007/s00586-023-07552-4).
