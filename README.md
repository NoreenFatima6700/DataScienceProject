
# Data Science Internship Project - DevelopersHub.co

## Overview
This project involves performing various data science tasks, including Exploratory Data Analysis (EDA), Sentiment Analysis, Fraud Detection, and House Price Prediction, utilizing datasets such as the Titanic dataset, IMDB reviews, Credit Card Fraud dataset, and the Boston Housing dataset.

## Tasks Completed

### Task 1: Exploratory Data Analysis (EDA)
- **Dataset Used**: Titanic Dataset
- **Steps Taken**:
  - Loaded the Titanic dataset using `Pandas`.
  - Cleaned the dataset by:
    - Handling missing values through imputation.
    - Removing duplicates.
    - Identifying and managing outliers using visualizations and statistical methods.
  - Created visualizations:
    - Bar charts for categorical variables (e.g., survived vs. not survived).
    - Histograms for numeric distributions (e.g., age, fare).
    - A correlation heatmap for numeric features.
  - Summarized insights, including observations about passenger survival rates, age distributions, and correlation between features like age, fare, and survival.

### Task 2: Text Sentiment Analysis
- **Dataset Used**: IMDB Reviews Dataset
- **Steps Taken**:
  - Preprocessed the text data by:
    - Tokenizing text into individual words.
    - Removing stopwords.
    - Performing lemmatization for text normalization.
  - Converted text data into numerical format using TF-IDF vectorization.
  - Trained a sentiment analysis model using:
    - **Logistic Regression**.
    - **Naive Bayes** classifier.
  - Evaluated model performance using **Precision**, **Recall**, and **F1-score** metrics.

### Task 3: Fraud Detection System
- **Dataset Used**: Credit Card Fraud Dataset
- **Steps Taken**:
  - Preprocessed the data by:
    - Handling imbalanced classes using **SMOTE** technique to generate synthetic data for fraudulent transactions.
  - Trained a **Random Forest** classifier to detect fraudulent transactions.
  - Evaluated the system using **Precision**, **Recall**, and **F1-score** metrics.
  - Created a simple command-line interface for testing the fraud detection system by inputting transaction details.

### Task 4: Predicting House Prices Using the Boston Housing Dataset
- **Dataset Used**: Boston Housing Dataset
- **Steps Taken**:
  - Preprocessed the data by:
    - Normalizing numerical features.
    - Handling missing values and preprocessing categorical variables.
  - Implemented **Linear Regression**, **Random Forest**, and **XGBoost** models from scratch (without using built-in functions like `sklearn.linear_model`).
  - Evaluated the models' performance using **RMSE** and **R²** metrics.
  - Compared the feature importance of **Random Forest** and **XGBoost** models and visualized the results.

## Requirements

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `sklearn`
  - `xgboost`
  - `imblearn` (for SMOTE)
  
You can install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

Where `requirements.txt` contains the following:
```
pandas
numpy
matplotlib
sklearn
xgboost
imblearn
```

## Usage

1. **Task 1 - EDA**:
   - Run the script for Task 1 to load the Titanic dataset, perform EDA, and generate visualizations.

2. **Task 2 - Sentiment Analysis**:
   - Run the sentiment analysis script to preprocess text data and build a sentiment classification model using logistic regression or Naive Bayes.

3. **Task 3 - Fraud Detection**:
   - Run the fraud detection system script to train and evaluate a Random Forest model for fraud detection.

4. **Task 4 - House Price Prediction**:
   - Run the regression script to predict house prices using the Boston Housing dataset with models like Linear Regression, Random Forest, and XGBoost.

## Next Steps

- Implement further model improvements, including hyperparameter tuning and cross-validation.
- Experiment with other classification and regression algorithms for enhanced results.
- Deploy models and develop web interfaces or APIs for easier interaction with the prediction systems.
