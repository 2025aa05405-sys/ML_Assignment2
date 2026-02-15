#!/usr/bin/env python3
"""Pre-compute confusion matrices and classification reports for cloud deployment."""

import pickle
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load test data
test_data = pd.read_csv('test_data.csv')
income_mapping = {0: '≤ 50K', 1: '> 50K'}

# Prepare test features and labels
if 'Income' in test_data.columns:
    X_test = test_data.drop('Income', axis=1)
    y_test = pd.Series([0 if '≤' in str(val) or '<=50K' in str(val) else 1 
                        for val in test_data['Income']])
else:
    print("Error: 'Income' column not found")
    exit(1)

# Load scaler and models
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X_test_scaled = scaler.transform(X_test)

# Model files
model_files = {
    'Logistic Regression': 'model/logistic_regression_model.pkl',
    'Decision Tree': 'model/decision_tree_model.pkl',
    'kNN': 'model/knn_model.pkl',
    'Naive Bayes': 'model/naive_bayes_model.pkl',
    'Random Forest': 'model/random_forest_model.pkl',
    'XGBoost': 'model/xgboost_model.pkl'
}

# Pre-compute for each model
predictions_data = {}
report_data = {}

for model_name, model_path in model_files.items():
    print(f"Processing {model_name}...")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                  target_names=['≤ 50K', '> 50K'],
                                  output_dict=True)
    
    # Save confusion matrix as image
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=['≤ 50K', '> 50K'],
               yticklabels=['≤ 50K', '> 50K'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f'model/confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Store predictions
    predictions_data[model_name] = {
        'confusion_matrix': cm.tolist(),
        'accuracy': float(report['accuracy'])
    }
    
    # Store report (convert numpy types to python types for JSON serialization)
    clean_report = {}
    for key, val in report.items():
        if isinstance(val, dict):
            clean_report[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in val.items()}
        else:
            clean_report[key] = float(val) if isinstance(val, (np.floating, np.integer)) else val
    
    report_data[model_name] = clean_report

# Save all data
with open('model/predictions_summary.json', 'w') as f:
    json.dump(predictions_data, f, indent=2)

with open('model/classification_reports.json', 'w') as f:
    json.dump(report_data, f, indent=2)

print("✓ All confusion matrices and reports pre-computed and saved!")
print("✓ Confusion matrix images saved in model/")
print("✓ JSON reports saved in model/classification_reports.json")
