import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="HAR Classification - ML Assignment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    models = {}
    model_files = [
        'logistic_regression_model.pkl',
        'decision_tree_model.pkl',
        'knn_model.pkl',
        'naive_bayes_model.pkl',
        'random_forest_model.pkl',
        'xgboost_model.pkl'
    ]
    
    for model_file in model_files:
        path = f'model/{model_file}'
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                models[model_name] = pickle.load(f)
    
    return models

@st.cache_resource
def load_scaler():
    with open('model/scaler.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_income_mapping():
    with open('model/income_mapping.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_results():
    if os.path.exists('model_results.csv'):
        return pd.read_csv('model_results.csv', index_col=0)
    return None

@st.cache_data
def load_test_data():
    if os.path.exists('test_data.csv'):
        return pd.read_csv('test_data.csv')
    return None

# Title and Description
st.title("💰 Adult Income Classification")
st.markdown("### ML Assignment: Binary Classification with Multiple Models")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["📊 Model Metrics", "🧪 Make Predictions", "📈 Model Comparison", "📥 Dataset Info"]
)

# Load all resources
models = load_models()
scaler = load_scaler()
income_mapping = load_income_mapping()
results_df = load_results()
test_data = load_test_data()

if page == "📊 Model Metrics":
    st.header("Model Performance Metrics")
    
    if results_df is not None:
        st.subheader("Evaluation Metrics Comparison")
        
        # Display results table
        st.dataframe(
            results_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn"),
            use_container_width=True
        )
        
        # Download results as CSV
        csv = results_df.to_csv()
        st.download_button(
            label="📥 Download Metrics (CSV)",
            data=csv,
            file_name="model_metrics.csv",
            mime="text/csv"
        )
        
        # Best models for each metric
        st.subheader("🏆 Best Models by Metric")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Accuracy", results_df['Accuracy'].idxmax(), 
                     f"{results_df['Accuracy'].max():.4f}")
            st.metric("Best F1 Score", results_df['F1'].idxmax(), 
                     f"{results_df['F1'].max():.4f}")
        
        with col2:
            st.metric("Best AUC", results_df['AUC'].idxmax(), 
                     f"{results_df['AUC'].max():.4f}")
            st.metric("Best Precision", results_df['Precision'].idxmax(), 
                     f"{results_df['Precision'].max():.4f}")
        
        with col3:
            st.metric("Best Recall", results_df['Recall'].idxmax(), 
                     f"{results_df['Recall'].max():.4f}")
            st.metric("Best MCC", results_df['MCC'].idxmax(), 
                     f"{results_df['MCC'].max():.4f}")
        
        # Visualization
        st.subheader("📉 Metrics Visualization")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Model Performance Across Metrics', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
        
        for idx, metric in enumerate(metrics):
            axes[idx].barh(results_df.index, results_df[metric], color=colors)
            axes[idx].set_xlabel(metric)
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_xlim(0, 1)
            for i, v in enumerate(results_df[metric]):
                axes[idx].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)


elif page == "🧪 Make Predictions":
    st.header("Test Model Predictions")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Select Model & Input Data")
        model_name = st.selectbox(
            "Choose a classification model:",
            list(models.keys()),
            index=0
        )
    
    with col2:
        input_method = st.radio("Input Method:", ["Upload CSV", "Sample from Test Data"])
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                user_data = pd.read_csv(uploaded_file)
                st.success(f"✓ File uploaded successfully! Shape: {user_data.shape}")
                
                if 'Activity' in user_data.columns:
                    X_user = user_data.drop('Activity', axis=1).values
                    y_user = user_data['Activity'].values
                else:
                    X_user = user_data.values
                    y_user = None
                
                # Scale the data
                X_user_scaled = scaler.transform(X_user)
                
                # Make predictions
                model = models[model_name]
                predictions = model.predict(X_user_scaled)
                probabilities = model.predict_proba(X_user_scaled)
                
                # Display results
                st.subheader("📊 Prediction Results")
                results_data = {
                    'Sample': range(len(predictions)),
                    'Predicted Income': [income_mapping.get(p, f"Class {p}") for p in predictions],
                    'Confidence': probabilities.max(axis=1)
                }
                
                if y_user is not None:
                    results_data['Actual Income'] = [income_mapping.get(p, f"Class {p}") for p in y_user]
                    results_data['Correct'] = predictions == y_user
                
                results_display_df = pd.DataFrame(results_data)
                st.dataframe(results_display_df, use_container_width=True)
                
                # Confusion Matrix if true labels exist
                if y_user is not None:
                    st.subheader("🔲 Confusion Matrix")
                    cm = confusion_matrix(y_user, predictions)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=[income_mapping.get(0, 'Class 0'), income_mapping.get(1, 'Class 1')],
                               yticklabels=[income_mapping.get(0, 'Class 0'), income_mapping.get(1, 'Class 1')])
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    ax.set_title(f'Confusion Matrix - {model_name}')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_user, predictions, 
                                                 target_names=[income_mapping.get(0, 'Class 0'), income_mapping.get(1, 'Class 1')],
                                                 output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                    
                    accuracy = accuracy_score(y_user, predictions)
                    st.info(f"✓ **Accuracy on Uploaded Data:** {accuracy:.4f}")
                
                # Download predictions
                csv = results_display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"predictions_{model_name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    else:  # Sample from Test Data
        if test_data is not None:
            st.subheader("Select Samples from Test Dataset")
            
            num_samples = st.slider("Number of samples to predict:", 1, min(20, len(test_data)), 5)
            sample_indices = st.multiselect(
                "Select sample indices:",
                range(min(10, len(test_data))),
                default=list(range(min(5, len(test_data))))
            )
            
            if sample_indices:
                sample_data = test_data.iloc[sample_indices].copy()
                
                if 'Income' in sample_data.columns:
                    X_sample = sample_data.drop('Income', axis=1).values
                    y_sample = sample_data['Income'].values
                else:
                    X_sample = sample_data.values
                    y_sample = None
                
                # Scale and predict
                X_sample_scaled = scaler.transform(X_sample)
                model = models[model_name]
                predictions = model.predict(X_sample_scaled)
                probabilities = model.predict_proba(X_sample_scaled)
                
                # Display results
                st.subheader("📊 Sample Predictions")
                for i, idx in enumerate(sample_indices):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Sample #{idx}**")
                        st.write(f"Predicted Income: **{income_mapping.get(predictions[i], f'Class {predictions[i]}')}**")
                        st.write(f"Confidence: **{probabilities[i].max():.4f}**")
                    
                    with col2:
                        if y_sample is not None:
                            actual = income_mapping.get(y_sample[i], f"Class {y_sample[i]}")
                            is_correct = predictions[i] == y_sample[i]
                            status = "✓ Correct" if is_correct else "✗ Incorrect"
                            st.write(f"Actual Income: **{actual}**")
                            st.write(f"Status: **{status}**")
                    
                    st.divider()


elif page == "📈 Model Comparison":
    st.header("Model Performance Comparison")
    
    if results_df is not None:
        # Radar chart comparison
        st.subheader("Radar Chart - Model Comparison")
        
        from math import pi
        
        categories = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
        
        for idx, (model_name, values) in enumerate(results_df.iterrows()):
            values_list = values[categories].tolist()
            values_list += values_list[:1]
            ax.plot(angles, values_list, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values_list, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Model Performance Radar Chart', size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Heatmap
        st.subheader("Heatmap - All Metrics")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(results_df, annot=True, fmt='.4f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1)
        ax.set_title('Model Performance Heatmap', fontweight='bold', fontsize=12)
        st.pyplot(fig)


elif page == "📥 Dataset Info":
    st.header("Dataset Information")
    
    st.subheader("Adult Income Census Dataset")
    
    st.markdown("""
    **Dataset Overview:**
    - **Source:** UCI Machine Learning Repository
    - **Target:** Binary Income Classification (≤ $50K, > $50K)
    - **Number of Instances:** 32,561 samples
    - **Number of Features:** 14 attributes
    - **Feature Types:** Continuous and Categorical
    
    **Description:**
    The Adult dataset is a benchmark dataset extracted from the 1994 U.S. Census database.
    It is commonly used for binary classification tasks to predict whether an individual's 
    income exceeds $50K/year based on demographic information.
    
    **Target Variable:**
    - **Class 0:** Income ≤ $50K (negative class)
    - **Class 1:** Income > $50K (positive class)
    
    **Features Include:**
    - Age, Education, Marital Status, Occupation, Relationship
    - Race, Sex, Capital Gain/Loss, Hours per Week, Native Country, etc.
    """)
    
    if test_data is not None:
        st.subheader("Test Dataset Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Samples", len(test_data))
        with col2:
            st.metric("Number of Features", len(test_data.columns) - 1)
        with col3:
            if 'Activity' in test_data.columns:
                st.metric("Number of Classes", test_data['Activity'].nunique())
        
        st.subheader("Feature Statistics")
        st.dataframe(test_data.describe(), use_container_width=True)
        
        st.subheader("Income Distribution")
        if 'Income' in test_data.columns:
            income_counts = test_data['Income'].value_counts().sort_index()
            income_labels = [income_mapping.get(int(i), f"Class {i}") for i in income_counts.index]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(income_labels, income_counts.values, color=plt.cm.Set3(np.linspace(0, 1, 2)))
            ax.set_ylabel('Count')
            ax.set_title('Income Distribution in Test Dataset')
            ax.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Download test data
        st.subheader("📥 Download Test Data")
        csv = test_data.to_csv(index=False)
        st.download_button(
            label="Download Test Data (CSV)",
            data=csv,
            file_name="test_data.csv",
            mime="text/csv"
        )
        
        st.info("Use this test data to evaluate the models with your own predictions!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>ML Classification Assignment - Adult Income Prediction</b></p>
    <p>Built with Streamlit | Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost</p>
</div>
""", unsafe_allow_html=True)
