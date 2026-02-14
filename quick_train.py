import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle
import os

# Create model directory
if not os.path.exists('model'):
    os.makedirs('model')

print("Creating Sample Dataset & Training Models...")

# Generate sample data for demonstration
np.random.seed(42)
n_samples_train = 8000
n_samples_test = 3000

# Create train data
X_train = np.random.randn(n_samples_train, 14)
y_train = np.random.binomial(1, 0.24, n_samples_train)

# Create test data  
X_test = np.random.randn(n_samples_test, 14)
y_test = np.random.binomial(1, 0.24, n_samples_test)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {}
results = {}

print("\nTraining Classification Models...")
print("="*60)

# 1. Logistic Regression
print("1. Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]
models['Logistic Regression'] = lr
results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_proba),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'MCC': matthews_corrcoef(y_test, y_pred)
}

# 2. Decision Tree
print("2. Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=15)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)[:, 1]
models['Decision Tree'] = dt
results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_proba),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'MCC': matthews_corrcoef(y_test, y_pred)
}

# 3. KNN
print("3. KNN...")
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]
models['KNN'] = knn
results['KNN'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_proba),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'MCC': matthews_corrcoef(y_test, y_pred)
}

# 4. Naive Bayes
print("4. Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_proba = nb.predict_proba(X_test)[:, 1]
models['Naive Bayes'] = nb
results['Naive Bayes'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_proba),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'MCC': matthews_corrcoef(y_test, y_pred)
}

# 5. Random Forest
print("5. Random Forest...")
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
models['Random Forest'] = rf
results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_proba),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'MCC': matthews_corrcoef(y_test, y_pred)
}

# 6. XGBoost
print("6. XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss', n_jobs=-1, use_label_encoder=False, max_depth=5)
xgb_model.fit(X_train, y_train, verbose=0)
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]
models['XGBoost'] = xgb_model
results['XGBoost'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_proba),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'MCC': matthews_corrcoef(y_test, y_pred)
}

# Display results
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
results_df = pd.DataFrame(results).T
print(results_df.to_string())

# Save results
results_df.to_csv('model_results.csv')
print("\n✓ Results saved to model_results.csv")

# Save models
print("\nSaving models...")
for model_name, model in models.items():
    path = f'model/{model_name.replace(" ", "_").lower()}_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ {path}")

# Save scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ model/scaler.pkl")

# Save mappings
income_mapping = {0: '<= 50K', 1: '> 50K'}
with open('model/income_mapping.pkl', 'wb') as f:
    pickle.dump(income_mapping, f)
print("✓ model/income_mapping.pkl")

# Create test dataset
test_df = pd.DataFrame(X_test, columns=[f'Feature_{i}' for i in range(14)])
test_df['Income'] = ['<= 50K' if y == 0 else '> 50K' for y in y_test]
test_df.to_csv('test_data.csv', index=False)
print("✓ test_data.csv created")

print("\n" + "="*60)
print("✓ Training Complete!")
print("="*60)
