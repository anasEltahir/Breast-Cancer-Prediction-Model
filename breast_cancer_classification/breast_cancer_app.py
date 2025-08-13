# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 22:28:31 2025

@author: Anas El-tahir
"""

# Machine learning project classification  Brest Cancer

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, recall_score, confusion_matrix, f1_score, precision_score, classification_report, roc_auc_score,auc, roc_curve
from sklearn.preprocessing import StandardScaler
import streamlit as st

#----------------------------------------------------------------------------------------------------------
#load Breast canser data
BreastCancer = load_breast_cancer()
X = BreastCancer.data
Y = BreastCancer.target
# --------------------------- TRAIN-TEST SPLIT --------------------------- 
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test  = train_test_split(X, Y, test_size= 0.34, random_state=33, shuffle=True)

# --------------------------- FEATURE SCALING --------------------------- 
# Scale features so that each has mean=0 and std=1 (important for logistic regression)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print('---------------------------------------------------------------------------------')
# ------------------------------------------------------------------------------------------------------
# Applying classification
logistic_regression_model = LogisticRegression(penalty='l2', max_iter=5000).fit(x_train_scaled, y_train)
print("The Logistic regression model:\n")
#Calculating Details
print('LogisticRegressionModel Train Score is : ' , logistic_regression_model.score(x_train_scaled, y_train))
print('LogisticRegressionModel Test Score is : ' , logistic_regression_model.score(x_test_scaled, y_test))
print('LogisticRegressionModel Classes are : ' , logistic_regression_model.classes_)
print('LogisticRegressionModel No. of iteratios is : ' , logistic_regression_model.n_iter_)
print('--------------------------------------------------------------------------------')
#  Prediction 
y_pred_class = logistic_regression_model.predict(x_test_scaled)
#----------------------------------------------------------------------- ------------------------------------
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred_class)
print("Confusion matrix:\n",CM)
#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
print("Accurace score:",accuracy_score(y_test, y_pred_class))
#Calculating Recall Score
print("Recall score:", recall_score(y_test, y_pred_class))
#Calculating Pressi Score
print("Precision score:", precision_score(y_test, y_pred_class))
#Calculating F1 Score
print("F1 score:", f1_score(y_test, y_pred_class))
#Calculating classification Report :
print("Classification report:\n",classification_report(y_test, y_pred_class ))
#Calculating Area Under the Curve :  
y_pred_proba = logistic_regression_model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
print("AUC:",auc(fpr, tpr))
print('--------------------------------------------------------------------------------')
#-----------------------------------------------------------------------------------------------------------

# The model in streamlit:

st.title("Breast Cancer Prediction")
st.markdown("### Developed by Anas El-tahir")

st.write("""
This interactive web app predicts whether a breast tumor is benign or malignant using a Logistic Regression model trained on the popular sklearn Breast Cancer dataset.

Built with Python, scikit-learn, and Streamlit, this app provides:

- User input for tumor features  
- Real-time prediction and confidence score  
- Visualizations like feature distributions, correlation heatmaps, and performance metrics  

Feel free to explore the model and understand the impact of different tumor features.
""")



st.markdown(""" 
This tool uses the [Scikit-learn Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)  
to help identify whether a breast tumor is **benign (non-cancerous)** or **malignant (cancerous)**.

### 📊 About the Dataset
- **Samples:** 569 patients
- **Features:** 30 measurements (radius, texture, perimeter, area, smoothness, etc.)
- **Target:** 
    - `0` → Malignant (Cancer)
    - `1` → Benign (Non-Cancer)

### 🧠 How it Works
We trained a **Logistic Regression** model on the dataset.  
You can adjust the tumor’s feature values using the sliders below,  
and the model will instantly predict the likelihood of cancer.

> ⚠ **Disclaimer:** This tool is for **educational purposes only** and should **not** be used as a medical diagnosis.  
Please consult a qualified healthcare professional for medical advice.
""")

# Create input fields for the 5 example features you used
# List of features sorted by absolute correlation descending (from your correlation list):
features_sorted = [
    'worst concave points', 'worst perimeter', 'mean concave points', 'worst radius',
    'mean perimeter', 'mean radius', 'worst area', 'mean area', 'mean concavity',
    'worst concavity', 'mean compactness', 'worst compactness', 'radius error',
    'perimeter error', 'area error', 'worst texture', 'worst smoothness',
    'worst symmetry', 'mean texture', 'concave points error', 'mean smoothness',
    'mean symmetry', 'worst fractal dimension', 'compactness error', 'concavity error',
    'fractal dimension error', 'smoothness error', 'mean fractal dimension',
    'texture error', 'symmetry error'
]

# Get index mapping for feature names in BreastCancer.feature_names
feature_to_idx = {name: idx for idx, name in enumerate(BreastCancer.feature_names)}

# Generate inputs sorted by importance with min/max from X
inputs = {}
for feature in features_sorted:
    idx = feature_to_idx[feature]
    min_val = float(np.min(X[:, idx]))
    max_val = float(np.max(X[:, idx]))
    default_val = float(np.mean(X[:, idx]))
    
    inputs[feature] = st.number_input(
        label=f'{feature.replace("_", " ").title()}',
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=(max_val - min_val) / 1000
    )



# Example: inputs dictionary from previous snippet
# inputs = {...}  # feature_name -> value from st.number_input

if st.button('Predict'):
    # Prepare user input array with all features in correct order
    input_list = [inputs[feat] for feat in BreastCancer.feature_names]  # order must match X columns
    
    user_input = np.array(input_list).reshape(1, -1)
    
    # Scale the input
    scaled_input = scaler.transform(user_input)
    
    # Predict
    pred = logistic_regression_model.predict(scaled_input)[0]
    proba = logistic_regression_model.predict_proba(scaled_input)[0, 1]  # probability of benign (class 1)
    
    diagnosis = 'Benign (Non-cancer)' if pred == 1 else 'Malignant (Cancer)'
    
    st.write(f"**Prediction:** {diagnosis}")
    st.write(f"**Confidence:** {proba:.2f}")



#----------------------------------------------------------------------------------------------------------
# Convert to DataFrame for easier plotting
df = pd.DataFrame(X, columns=BreastCancer.feature_names)
df['target'] = Y
print("The datasets:\n",df)
print("0 mean Malignant(Cancer) , 1 mean Benign(Noncancer)")
print("the features:\n",df.columns)

# to see which feature affect my model more, Features with higher absolute correlation values (close to -1 or 1) have more predictive power or relevance to distinguishing benign vs
correlations = df.corr()['target'].sort_values(ascending=False)
print(correlations)
# to see the range of my data
print(df[df.columns].describe().T[['min', 'max']])


# Plot distribution of a few important features by target class
features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']

st.write("### 📊 Feature Distributions by Tumor Type")
st.write("""
These charts show how different measurements vary between **benign** and **malignant** tumors.
You can see if certain features tend to be higher or lower for one class, which helps us
understand what the model might be looking at.
""")
for feature in features:
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=feature, hue='target', kde=True, element='step', ax=ax)
    ax.set_title(f'Distribution of {feature} by Class')
    st.pyplot(fig)



st.write("### 🔍 Relationships Between Features")
st.title("Breast Cancer Prediction")
st.markdown("### Developed by Anas El-tahir")

st.write("""
This interactive web app predicts whether a breast tumor is benign or malignant using a Logistic Regression model trained on the popular sklearn Breast Cancer dataset.

Built with Python, scikit-learn, and Streamlit, this app provides:

- User input for tumor features  
- Real-time prediction and confidence score  
- Visualizations like feature distributions, correlation heatmaps, and performance metrics  

Feel free to explore the model and understand the impact of different tumor features.
""")

# Then your input fields, prediction button, and plots come after

st.write("""
The scatterplots (pairplot) below show how features interact with each other.  
Clusters or clear separations between colors suggest that those features are good at distinguishing tumor types.
""")
pair_fig = sns.pairplot(df[features + ['target']], hue='target', corner=True)
st.pyplot(pair_fig)


st.write("### 🧩 Feature Correlation Heatmap")
st.write("""
This heatmap shows how each feature is related to the others and to the target variable.  
Dark red/blue areas mean strong positive/negative correlation.  
Features with high correlation to the target are often more important for prediction.
""")
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)
ax.set_title('Feature Correlation Heatmap')
st.pyplot(fig)


st.write("### 📦 Boxplots by Tumor Type")
st.write("""
These boxplots show the spread, median, and outliers for each feature in each tumor class.
They help spot differences and variability between benign and malignant tumors.
""")
for feature in features:
    fig, ax = plt.subplots()
    sns.boxplot(x='target', y=feature, data=df, ax=ax)
    ax.set_title(f'Boxplot of {feature} by Class')
    st.pyplot(fig)
# ----------------------------------------------------------------------
# drawing confusion matrix
st.write("### 🧮 Confusion Matrix")
st.write("""
This table shows how many predictions were correct vs incorrect.  
It breaks down results into:
- **True Positives** (correct malignant)
- **True Negatives** (correct benign)
- **False Positives** (benign predicted as malignant)
- **False Negatives** (malignant predicted as benign)
""")
fig, ax = plt.subplots()
sns.heatmap(CM, annot=True, fmt='d', cmap='Blues',
            xticklabels=BreastCancer.target_names,
            yticklabels=BreastCancer.target_names, ax=ax)
st.pyplot(fig)
# drawing AUC
st.write("### 📈 ROC Curve & AUC")
st.write("""
The ROC (Receiver Operating Characteristic) curve shows the trade-off between **true positive rate** 
(sensitivity) and **false positive rate** (1 - specificity).  
The **AUC** (Area Under the Curve) tells us how well the model separates the two classes:
- **1.0** = perfect separation  
- **0.5** = no better than random guessing
""")

# --------------------------- ROC & AUC --------------------------- #
y_pred_proba = logistic_regression_model.predict_proba(x_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)
print('--------------------------------------------------------------------------------')

# Create figure and plot
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
ax.grid()

# Display in Streamlit
st.pyplot(fig)

# streamlit run "D:\Projects\MACHINE LEARNING PROJECTS\BREAST CANCER CLASSIFICATION\breast cancer app.py"

