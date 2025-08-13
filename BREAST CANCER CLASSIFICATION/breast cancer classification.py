# -*- coding: utf-8 -*-
"""
Created on Sat Aug 9 22:28:31 2025
@author: Anas El-tahir

PROJECT TITLE: Breast Cancer Classification using Logistic Regression

DESCRIPTION:
This project applies machine learning classification techniques to the
Breast Cancer Wisconsin dataset, using Logistic Regression to predict 
whether a tumor is malignant (cancer) or benign (non-cancer).

STEPS:
1. Load and explore the dataset, including statistical summaries and visualizations.
2. Analyze correlations between features and the target to identify 
   the most relevant predictors.
3. Preprocess the data by splitting into training/testing sets and scaling features.
4. Train a Logistic Regression classifier.
5. Evaluate the model using Accuracy, Recall, Precision, F1-score, 
   Confusion Matrix, and ROC-AUC.
6. Visualize results to understand the model's strengths and weaknesses.

GOAL:
To build a classification model with strong recall for malignant tumors,
ensuring that potential cancer cases are correctly identified.
"""

# Machine learning project --> classification -->  Breast Cancer

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import  accuracy_score, recall_score, confusion_matrix, f1_score, precision_score, classification_report, roc_auc_score,auc, roc_curve
from sklearn.preprocessing import StandardScaler

# --------------------------- LOAD DATASET ---------------------------
#load Breast cancer data
BreastCancer = load_breast_cancer()

X = BreastCancer.data
Y = BreastCancer.target
# --------------------------- CREATE DATAFRAME --------------------------- 
# Convert to DataFrame for easier inspection & plotting
df = pd.DataFrame(X, columns=BreastCancer.feature_names)
df['target'] = Y
print("The datasets:\n",df)
print("0 mean Malignant(Cancer) , 1 mean Benign(Noncancer)")
print("the features:\n",df.columns)

# --------------------------- FEATURE CORRELATION ANALYSIS --------------------------- 
# Identify which features have the strongest relationship with the target
correlations = df.corr()['target'].sort_values(ascending=False)
print("\nFeature correlations with target (values closer to ±1 indicate stronger relationships):\n", correlations)
# Show min and max of each feature  # to see the range of the data
print(df[df.columns].describe().T[['min', 'max']])

# --------------------------- DATA VISUALIZATION --------------------------- 
# Select a few important features for visualization
features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
# Distribution plots by class
print("Shows some feature distribution for per - class:\n")
plt.figure(figsize=(15,10))
for i, feature in enumerate(features):
    plt.subplot(3, 2, i+1)
    sns.histplot(data=df, x=feature, hue='target', kde=True, element='step')
    plt.title(f'Distribution of {feature} by Class')
plt.tight_layout()
plt.show()
# Pairplot to visualize relationships between selected features
print("Scatterplots between features + histograms to Help see patterns between multiple features")
sns.pairplot(df[features + ['target']], hue='target', corner=True)
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()
# Heatmap of correlations
print("Shows correlation between features, Useful for spotting highly related features:\n")
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Feature Correlation Heatmap')
plt.show()
# Boxplots for distribution spread & outliers
print("Shows spread, median, and outliers of some features by class.")
plt.figure(figsize=(15,10))
for i, feature in enumerate(features):
    plt.subplot(3, 2, i+1)
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Class')
plt.tight_layout()
plt.show()
# --------------------------- TRAIN-TEST SPLIT --------------------------- 
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test  = train_test_split(X, Y, test_size= 0.34, random_state=33, shuffle=True)

# --------------------------- FEATURE SCALING --------------------------- 
# Scale features so that each has mean=0 and std=1 (important for logistic regression)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
#----------------------------------------------------------------------------------------------------------


print('---------------------------------------------------------------------------------')
# --------------------------- TRAIN LOGISTIC REGRESSION MODEL --------------------------- 

logistic_regression_model = LogisticRegression(penalty='l2', max_iter=5000).fit(x_train_scaled, y_train)
print("The Logistic regression model:\n")
#Calculating Details
print('LogisticRegressionModel Train Score is : ' , logistic_regression_model.score(x_train_scaled, y_train))
print('LogisticRegressionModel Test Score is : ' , logistic_regression_model.score(x_test_scaled, y_test))
print('LogisticRegressionModel Classes are : ' , logistic_regression_model.classes_)
print('LogisticRegressionModel No. of iteratios is : ' , logistic_regression_model.n_iter_)
print('--------------------------------------------------------------------------------')
# --------------------------- PREDICTION --------------------------- 

y_pred_class = logistic_regression_model.predict(x_test_scaled)
# print(y_test[:30])
# print(y_pred_class[:30])
#----------------------------------------------------------------------- ------------------------------------
# --------------------------- CONFUSION MATRIX --------------------------- 
CM = confusion_matrix(y_test, y_pred_class)
print("Confusion matrix:\n",CM)
# drawing confusion matrix
sns.heatmap(CM, annot=True, fmt='d', cmap='Blues',
            xticklabels=BreastCancer.target_names,
            yticklabels=BreastCancer.target_names)
plt.show()

# --------------------------- EVALUATION METRICS --------------------------- 

#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
print("Accuracy score:",accuracy_score(y_test, y_pred_class))
#Calculating Recall Score
print("Recall score:", recall_score(y_test, y_pred_class))
#Calculating Precision Score
print("Precision score:", precision_score(y_test, y_pred_class))
#Calculating F1 Score
print("F1 score:", f1_score(y_test, y_pred_class))
#Calculating classification Report :
print("Classification report:\n",classification_report(y_test, y_pred_class ))
# --------------------------- ROC & AUC --------------------------- #
#Calculating Area Under the Curve :  
y_pred_proba = logistic_regression_model.predict_proba(x_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
print("AUC:",auc(fpr, tpr))
print('--------------------------------------------------------------------------------')
plt.figure(figsize=(8,8))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc(fpr, tpr):.3f})")
plt.xlabel("False positive rate" )
plt.ylabel("True positive rate" )
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# --------------------------- INTERPRETATION --------------------------- 

print("\nInterpretation:")
print("High recall means the model is effective at detecting malignant tumors,")
print("which is critical in medical screening where missing a cancer case can be costly.")





