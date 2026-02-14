import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
import warnings
import sys
warnings.filterwarnings('ignore')

# Create model directory
if not os.path.exists('model'):
    os.makedirs('model')

# Download and load Adult dataset
print("Loading Adult Income Dataset from UCI Repository...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

try:
    print("Downloading training data...")
    sys.stdout.flush()
    X_train_raw = pd.read_csv(url, header=None, names=column_names, sep=', ', engine='python')
    y_train_raw = X_train_raw.pop('income')
    
    print("Downloading test data...")
    sys.stdout.flush()
    X_test_raw = pd.read_csv(url_test, header=None, names=column_names, sep=', ', engine='python', skiprows=1)
    y_test_raw = X_test_raw.pop('income')
    
    # Clean up target variable - remove dots from test set labels
    y_test_raw = y_test_raw.str.replace('.', '', regex=False).str.strip()
    y_train_raw = y_train_raw.str.strip()
    
except Exception as e:
    print(f"Error downloading: {e}")
    print("Using sample data instead...")
    sys.stdout.flush()
    
    # Create sample data locally for demonstration
    np.random.seed(42)
    n_samples_train = 15000
    n_samples_test = 5000
    
    X_train_raw = pd.DataFrame({
        'age': np.random.randint(17, 90, n_samples_train),
        'workclass': np.random.choice(['Private', 'Self-emp', 'Government', 'Other', ' ?'], n_samples_train),
        'fnlwgt': np.random.randint(12000, 1500000, n_samples_train),
        'education': np.random.choice(['HS-grad', 'Bachelors', 'Masters', 'Doctorate', 'Some-college'], n_samples_train),
        'education-num': np.random.randint(1, 17, n_samples_train),
        'marital-status': np.random.choice(['Married-civ-spouse', 'Divorced', 'Single', 'Widowed'], n_samples_train),
        'occupation': np.random.choice(['Tech', 'Sales', 'Exec-managerial', 'Craft-repair', ' ?'], n_samples_train),
        'relationship': np.random.choice(['Husband', 'Wife', 'Son', 'Daughter', 'Other'], n_samples_train),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Other', 'Amer-Indian'], n_samples_train),
        'sex': np.random.choice(['Male', 'Female'], n_samples_train),
        'capital-gain': np.random.randint(0, 100000, n_samples_train),
        'capital-loss': np.random.randint(0, 5000, n_samples_train),
        'hours-per-week': np.random.randint(1, 100, n_samples_train),
        'native-country': np.random.choice(['United-States', 'Other', 'Mexico', 'India', 'China'], n_samples_train)
    })
    y_train_raw = pd.Series(np.random.choice([' <=50K', ' >50K'], n_samples_train, p=[0.76, 0.24]))
    
    X_test_raw = pd.DataFrame({
        'age': np.random.randint(17, 90, n_samples_test),
        'workclass': np.random.choice(['Private', 'Self-emp', 'Government', 'Other', ' ?'], n_samples_test),
        'fnlwgt': np.random.randint(12000, 1500000, n_samples_test),
        'education': np.random.choice(['HS-grad', 'Bachelors', 'Masters', 'Doctorate', 'Some-college'], n_samples_test),
        'education-num': np.random.randint(1, 17, n_samples_test),
        'marital-status': np.random.choice(['Married-civ-spouse', 'Divorced', 'Single', 'Widowed'], n_samples_test),
        'occupation': np.random.choice(['Tech', 'Sales', 'Exec-managerial', 'Craft-repair', ' ?'], n_samples_test),
        'relationship': np.random.choice(['Husband', 'Wife', 'Son', 'Daughter', 'Other'], n_samples_test),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Other', 'Amer-Indian'], n_samples_test),
        'sex': np.random.choice(['Male', 'Female'], n_samples_test),
        'capital-gain': np.random.randint(0, 100000, n_samples_test),
        'capital-loss': np.random.randint(0, 5000, n_samples_test),
        'hours-per-week': np.random.randint(1, 100, n_samples_test),
        'native-country': np.random.choice(['United-States', 'Other', 'Mexico', 'India', 'China'], n_samples_test)
    })
    y_test_raw = pd.Series(np.random.choice([' <=50K.', ' >50K.'], n_samples_test, p=[0.76, 0.24]))
    y_test_raw = y_test_raw.str.replace('.', '', regex=False).str.strip()

print(f"Training set shape: {X_train_raw.shape}")
print(f"Test set shape: {X_test_raw.shape}")
sys.stdout.flush()

# Combine for preprocessing
X = pd.concat([X_train_raw, X_test_raw], ignore_index=True)
y = pd.concat([y_train_raw, y_test_raw], ignore_index=True)

# Handle missing values
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].str.strip() if isinstance(X[col].iloc[0], str) else X[col]
        X[col] = X[col].replace('?', np.nan)

# Fill missing values with mode
for col in X.select_dtypes(include=['object']).columns:
    mode_val = X[col].mode()
    if len(mode_val) > 0:
        X[col].fillna(mode_val[0], inplace=True)

for col in X.select_dtypes(include=['float64', 'int64']).columns:
    X[col].fillna(X[col].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(y.values)

# Split back
X_train_split = X.iloc[:len(X_train_raw)].values.astype(float)
X_test_split = X.iloc[len(X_train_raw):].values.astype(float)
y_train_array = y_encoded[:len(X_train_raw)]
y_test_array = y_encoded[len(X_train_raw):]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_test_scaled = scaler.transform(X_test_split)

# Save test data for Streamlit app
test_df = pd.DataFrame(X_test_scaled, columns=[f'Feature_{i}' for i in range(X_test_scaled.shape[1])])
test_df['Income'] = y_encoder.inverse_transform(y_test_array)
test_df.to_csv('test_data.csv', index=False)
print("✓ Test data saved to test_data.csv")
sys.stdout.flush()

# Dictionary to store models and results
models = {}
results = {}

print("\n" + "="*80)
print("Training Classification Models...")
print("="*80)

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
sys.stdout.flush()
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_array)
y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
models['Logistic Regression'] = lr

results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test_array, y_pred_lr),
    'AUC': roc_auc_score(y_test_array, y_pred_proba_lr),
    'Precision': precision_score(y_test_array, y_pred_lr, zero_division=0),
    'Recall': recall_score(y_test_array, y_pred_lr, zero_division=0),
    'F1': f1_score(y_test_array, y_pred_lr, zero_division=0),
    'MCC': matthews_corrcoef(y_test_array, y_pred_lr)
}

# 2. Decision Tree Classifier
print("2. Training Decision Tree Classifier...")
sys.stdout.flush()
dt = DecisionTreeClassifier(random_state=42, max_depth=15)
dt.fit(X_train_scaled, y_train_array)
y_pred_dt = dt.predict(X_test_scaled)
y_pred_proba_dt = dt.predict_proba(X_test_scaled)[:, 1]
models['Decision Tree'] = dt

results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test_array, y_pred_dt),
    'AUC': roc_auc_score(y_test_array, y_pred_proba_dt),
    'Precision': precision_score(y_test_array, y_pred_dt, zero_division=0),
    'Recall': recall_score(y_test_array, y_pred_dt, zero_division=0),
    'F1': f1_score(y_test_array, y_pred_dt, zero_division=0),
    'MCC': matthews_corrcoef(y_test_array, y_pred_dt)
}

# 3. K-Nearest Neighbor Classifier
print("3. Training K-Nearest Neighbor Classifier...")
sys.stdout.flush()
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_scaled, y_train_array)
y_pred_knn = knn.predict(X_test_scaled)
y_pred_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]
models['KNN'] = knn

results['KNN'] = {
    'Accuracy': accuracy_score(y_test_array, y_pred_knn),
    'AUC': roc_auc_score(y_test_array, y_pred_proba_knn),
    'Precision': precision_score(y_test_array, y_pred_knn, zero_division=0),
    'Recall': recall_score(y_test_array, y_pred_knn, zero_division=0),
    'F1': f1_score(y_test_array, y_pred_knn, zero_division=0),
    'MCC': matthews_corrcoef(y_test_array, y_pred_knn)
}

# 4. Naive Bayes Classifier (Gaussian)
print("4. Training Naive Bayes Classifier...")
sys.stdout.flush()
nb = GaussianNB()
nb.fit(X_train_scaled, y_train_array)
y_pred_nb = nb.predict(X_test_scaled)
y_pred_proba_nb = nb.predict_proba(X_test_scaled)[:, 1]
models['Naive Bayes'] = nb

results['Naive Bayes'] = {
    'Accuracy': accuracy_score(y_test_array, y_pred_nb),
    'AUC': roc_auc_score(y_test_array, y_pred_proba_nb),
    'Precision': precision_score(y_test_array, y_pred_nb, zero_division=0),
    'Recall': recall_score(y_test_array, y_pred_nb, zero_division=0),
    'F1': f1_score(y_test_array, y_pred_nb, zero_division=0),
    'MCC': matthews_corrcoef(y_test_array, y_pred_nb)
}

# 5. Random Forest Classifier
print("5. Training Random Forest Classifier...")
sys.stdout.flush()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
rf.fit(X_train_scaled, y_train_array)
y_pred_rf = rf.predict(X_test_scaled)
y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
models['Random Forest'] = rf

results['Random Forest'] = {
    'Accuracy': accuracy_score(y_test_array, y_pred_rf),
    'AUC': roc_auc_score(y_test_array, y_pred_proba_rf),
    'Precision': precision_score(y_test_array, y_pred_rf, zero_division=0),
    'Recall': recall_score(y_test_array, y_pred_rf, zero_division=0),
    'F1': f1_score(y_test_array, y_pred_rf, zero_division=0),
    'MCC': matthews_corrcoef(y_test_array, y_pred_rf)
}

# 6. XGBoost Classifier
print("6. Training XGBoost Classifier...")
sys.stdout.flush()
xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss', n_jobs=-1, use_label_encoder=False, max_depth=5)
xgb_model.fit(X_train_scaled, y_train_array, verbose=0)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
models['XGBoost'] = xgb_model

results['XGBoost'] = {
    'Accuracy': accuracy_score(y_test_array, y_pred_xgb),
    'AUC': roc_auc_score(y_test_array, y_pred_proba_xgb),
    'Precision': precision_score(y_test_array, y_pred_xgb, zero_division=0),
    'Recall': recall_score(y_test_array, y_pred_xgb, zero_division=0),
    'F1': f1_score(y_test_array, y_pred_xgb, zero_division=0),
    'MCC': matthews_corrcoef(y_test_array, y_pred_xgb)
}

# Print results
print("\n" + "="*80)
print("MODEL EVALUATION RESULTS")
print("="*80)

results_df = pd.DataFrame(results).T
print("\n", results_df)
sys.stdout.flush()

# Save results to CSV
results_df.to_csv('model_results.csv')
print("\n✓ Results saved to model_results.csv")
sys.stdout.flush()

# Save models
print("\nSaving models...")
sys.stdout.flush()
for model_name, model in models.items():
    model_path = f'model/{model_name.replace(" ", "_").lower()}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved: {model_path}")
    sys.stdout.flush()

# Save scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: model/scaler.pkl")
sys.stdout.flush()

# Save income mapping
income_mapping = {
    0: '<= 50K',
    1: '> 50K'
}

with open('model/income_mapping.pkl', 'wb') as f:
    pickle.dump(income_mapping, f)
print("✓ Saved: model/income_mapping.pkl")
sys.stdout.flush()

# Save label encoders for feature columns
with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ Saved: model/label_encoders.pkl")
sys.stdout.flush()

# Save target encoder
with open('model/target_encoder.pkl', 'wb') as f:
    pickle.dump(y_encoder, f)
print("✓ Saved: model/target_encoder.pkl")
sys.stdout.flush()

print("\n" + "="*80)
print("Training Complete!")
print("="*80)
