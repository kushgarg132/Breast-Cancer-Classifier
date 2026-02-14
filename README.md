# ML Assignment 2 - Breast Cancer Classification

## Live Demo
Check out the live application here: [Breast Cancer Classifier](https://breast-cancer-classifier-hnxzqbpdg9eg896xgmsczf.streamlit.app/)

## Problem Statement
The goal of this assignment is to implement multiple machine learning classification models to predict whether a breast mass is benign or malignant. The project involves analyzing the Breast Cancer Wisconsin (Diagnostic) dataset, training 6 different models, evaluating their performance, and deploying a Streamlit web application for interactive predictions.

## Dataset Description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** sklearn.datasets
- **Features:** 30 numeric features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
- **Classes:** Malignant (0), Benign (1)
- **Samples:** 569 instances

## Models Used & Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| Decision Tree | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| kNN | 0.9474 | 0.9820 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| Naive Bayes | 0.9737 | 0.9984 | 0.9595 | 1.0000 | 0.9793 | 0.9447 |
| Random Forest (Ensemble) | 0.9649 | 0.9953 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| XGBoost (Ensemble) | 0.9561 | 0.9908 | 0.9583 | 0.9718 | 0.9650 | 0.9064 |

### Observations
1. **Naive Bayes** achieved the highest **Recall (1.0000)** and **AUC (0.9984)**. In the context of cancer diagnosis, high recall is crucial to minimize false negatives (missing a malignant case). Thus, Naive Bayes is a strong candidate for this problem.
2. **Logistic Regression** tied with Naive Bayes for the highest **Accuracy (0.9737)** and performed consistently well across all metrics.
3. **Random Forest** and **XGBoost** also performed robustly, showing high accuracy (>95%), but were slightly outperformed by the simpler linear and probabilistic models on this specific test split.
4. **Decision Tree** and **KNN** had the lowest accuracy (94.74%) but are still very effective classifiers for this dataset.

## How to Run locally

1. **Install Dependencies:**
   Since the environment may have restrictions, a virtual environment is recommended.
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models:**
   ```bash
   python train_models.py
   ```
   This will save the trained models in the `model/` directory and print the performance metrics.

3. **Run Streamlit App:**
   ```bash
   streamlit run app.py
   ```
   Upload the `model/test_data.csv` file to test the predictions.
