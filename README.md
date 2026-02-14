# 🏦 Bank Marketing Term Deposit Prediction

This project is my implementation of **Machine Learning Assignment 2** using the
**Bank Marketing** dataset from the UCI Machine Learning Repository.[web:39] It
covers end-to-end model building, evaluation, and deployment using a Streamlit
web app, as required in the assignment instructions.[file:1]

---

## 1. Problem statement

The Portuguese banking institution regularly runs direct marketing campaigns by
phone to promote long-term term deposit products to its clients.[web:39][web:54]
Traditionally, deciding which clients to call and evaluating the success of
these campaigns relies heavily on human judgment, which can be inefficient and
inconsistent.[web:54][web:49] The goal of this project is to build and compare
multiple machine learning classification models that predict whether a client
will subscribe to a term deposit based on their profile, past contact history,
and macro-economic indicators.[web:39][web:54] These models can help the bank
prioritise high-probability clients, improve campaign efficiency, and reduce
unnecessary calls to clients who are unlikely to subscribe.[web:39][web:54]

---

## 2. Dataset description

### Source

The dataset used in this project is the **Bank Marketing** dataset from the
**UCI Machine Learning Repository**.[web:36][web:39] It contains data from
phone-based direct marketing campaigns run by a Portuguese bank to promote term
deposits.[web:39][web:54]

### Size and structure

In this implementation I use the `bank-additional-full.csv` file, which has:

- **41,188 instances (rows)**.
- **20 input attributes** plus **1 target variable**.[web:39][web:54]

This satisfies the assignment constraints of at least **12 features** and **500
instances**.[file:1]

### Input features

The input variables fall into three groups:[web:39][web:54]

- **Client information**

  - `age`: Client's age (numeric).[web:54]
  - `job`: Type of job (admin., blue-collar, technician, services, etc.).[web:39][web:54]
  - `marital`: Marital status (married, single, divorced, etc.).[web:39][web:54]
  - `education`: Education level (basic, high school, university degree,
    etc.).[web:39][web:54]
  - `default`: Has credit in default? (yes, no, unknown).[web:39][web:54]
  - `housing`: Has housing loan? (yes, no, unknown).[web:39][web:54]
  - `loan`: Has personal loan? (yes, no, unknown).[web:39][web:54]

- **Current campaign and contact**

  - `contact`: Contact communication type (cellular, telephone).[web:39][web:54]
  - `month`: Last contact month of year (jan, feb, …, dec).[web:39][web:54]
  - `day_of_week`: Last contact day of the week (mon, tue, …).[web:39][web:54]
  - `duration`: Last contact duration in seconds (numeric).[web:39][web:54]
  - `campaign`: Number of contacts performed during this campaign.[web:39][web:54]
  - `pdays`: Days since last contact in a previous campaign (numeric; 999 means
    not previously contacted).[web:39][web:54]
  - `previous`: Number of contacts before this campaign.[web:39][web:54]
  - `poutcome`: Outcome of the previous marketing campaign (success, failure,
    non-existent, etc.).[web:39][web:54]

- **Socio-economic indicators**

  - `emp.var.rate`: Employment variation rate (quarterly indicator).[web:39][web:54]
  - `cons.price.idx`: Consumer price index (monthly).[web:39][web:54]
  - `cons.conf.idx`: Consumer confidence index (monthly).[web:39][web:54]
  - `euribor3m`: Euribor 3-month rate.[web:39][web:54]
  - `nr.employed`: Number of employees (quarterly).[web:39][web:54]

### Target variable

- `y`: Whether the client subscribed to a term deposit at the end of the
  campaign (binary: **"yes"** or **"no"**).[web:39][web:54] In the modelling
  code this is converted to **1** for "yes" and **0** for "no".

### Preprocessing

- Categorical features (such as `job`, `marital`, `education`, `default`,
  `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`) are encoded
  using one-hot encoding.
- Numeric features (such as `age`, `duration`, `campaign`, `pdays`, `previous`,
  `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`,
  `nr.employed`) are standardised using a scaler.
- The dataset is split into training and test sets using a stratified
  train/test split so that the proportion of clients who subscribed vs. those
  who did not is preserved in both splits.

---

## 3. Models used and evaluation metrics

All six models listed in the assignment are implemented on the same dataset:[file:1]

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbour (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (ensemble)  
6. XGBoost (ensemble)

For each model, the following metrics are calculated on the test set:

- Accuracy  
- AUC (ROC AUC)  
- Precision  
- Recall  
- F1 score  
- Matthews Correlation Coefficient (MCC)[file:1]

After running `model_training.py`, the metrics are saved to `model/model_results.csv`.


```markdown
| ML Model Name       | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|---------------------|----------|------|-----------|--------|------|------|
| Logistic Regression |0.9166    |0.9424|0.7118     |0.4364  |0.5411|0.5162|
| Decision Tree       |0.9144    |0.8893|0.6493     |0.5226  |0.5791|0.5360|
| kNN                 |0.9076    |0.8799|0.6259     |0.4472  |0.5217|0.4803|
| Naive Bayes         |0.8203    |0.8393|0.3495     |0.6907  |0.4642|0.4009|
| Random Forest       |0.9121    |0.9442|0.7849     |0.3028  |0.4370|0.4532|
| XGBoost             |0.9245    |0.9548|0.7008     |0.5754  |0.6320|0.5939|
