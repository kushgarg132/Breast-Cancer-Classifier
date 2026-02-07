import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef

# Set Page Config
st.set_page_config(page_title="ML Assignment 2", layout="wide")

# Title and Description
st.title("Breast Cancer Classification App")
st.markdown("""
This app predicts whether a breast mass is benign or malignant using various machine learning models.
Please upload a CSV file with the required features.
""")

# Sidebar
st.sidebar.header("User Input Features")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Model Selection
model_name = st.sidebar.selectbox("Select Model", 
    ("Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"))

# Load Models (Cached)
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
        "XGBoost": "model/xgboost.pkl"
    }
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.error(f"Error loading {name}: {e}")
    return models

# Load Scaler
@st.cache_resource
def load_scaler():
    try:
        return joblib.load('model/scaler.pkl')
    except:
        return None

models = load_models()
scaler = load_scaler()

# Main Logic
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data (First 5 Rows)")
        st.write(input_df.head())
        
        # Check for target column
        if 'target' in input_df.columns:
            y_true = input_df['target']
            X = input_df.drop('target', axis=1)
        else:
            y_true = None
            X = input_df
            
        # Scale data if needed
        if model_name in ["Logistic Regression", "KNN"] and scaler is not None:
            # Check if feature count matches
            # Note: StandardScaler check is strict, but we'll try-catch or check shape
            if X.shape[1] == scaler.n_features_in_:
                 X_processed = scaler.transform(X)
            else:
                 st.warning(f"Feature count mismatch. Scaler expects {scaler.n_features_in_}, got {X.shape[1]}. Using raw data (might affect performance).")
                 X_processed = X
        else:
            X_processed = X
            
        # Prediction
        if model_name in models:
            model = models[model_name]
            
            # Ensure columns match training data if possible, or just predict (sklearn assumes order)
            try:
                prediction = model.predict(X_processed)
                if hasattr(model, "predict_proba"):
                    prediction_proba = model.predict_proba(X_processed)
                else:
                    prediction_proba = None
                    
                st.subheader("Prediction Results")
                results_df = X.copy()
                results_df['Prediction'] = prediction
                st.write(results_df)
                
                # Metrics
                if y_true is not None:
                    st.subheader("Evaluation Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{accuracy_score(y_true, prediction):.4f}")
                    col1.metric("Precision", f"{precision_score(y_true, prediction, average='weighted', zero_division=0):.4f}")
                    col2.metric("Recall", f"{recall_score(y_true, prediction, average='weighted', zero_division=0):.4f}")
                    col2.metric("F1 Score", f"{f1_score(y_true, prediction, average='weighted', zero_division=0):.4f}")
                    col3.metric("MCC", f"{matthews_corrcoef(y_true, prediction):.4f}")
                    
                    if prediction_proba is not None:
                         # AUC
                         try:
                             if len(np.unique(y_true)) > 2:
                                 auc = roc_auc_score(y_true, prediction_proba, multi_class='ovr')
                             else:
                                 # For binary, need to check shape or just take column 1
                                 if prediction_proba.shape[1] == 2:
                                    auc = roc_auc_score(y_true, prediction_proba[:, 1])
                                 else:
                                    auc = roc_auc_score(y_true, prediction_proba[:, 1]) # Attempt
                             col3.metric("AUC Score", f"{auc:.4f}")
                         except Exception as e:
                             col3.metric("AUC Score", "N/A")
                             # st.warning(f"AUC Calc Error: {e}")
                    
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_true, prediction)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig)
                    
                    st.subheader("Classification Report")
                    st.text(classification_report(y_true, prediction))
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                
        else:
             st.error("Model not loaded correctly.")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed. You can use the `model/test_data.csv` generated by the training script.")
