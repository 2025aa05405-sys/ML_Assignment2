import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page
st.set_page_config(
    page_title="Adult Income Classification - ML Assignment",
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
    # Models are not loaded at runtime; results are pre-computed
    return {}

@st.cache_resource
def load_scaler():
    # Scaler not needed; using pre-computed results
    return None

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
    st.info("ℹ️ **Static Predictions (Pre-Computed)**\n\nModels have been trained locally. Predictions are displayed from pre-computed results to ensure fast cloud deployment.\n\nTo test custom predictions, download the test data and run the models locally using the repository code.")


elif page == "📈 Model Comparison":
    st.header("Model Performance Comparison")
    
    if results_df is not None:
        # Bar chart comparison
        st.subheader("Model Performance Metrics")
        
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
            if 'Income' in test_data.columns:
                st.metric("Number of Classes", test_data['Income'].nunique())
        
        st.subheader("Feature Statistics")
        st.dataframe(test_data.describe(), use_container_width=True)
        
        st.subheader("Income Distribution")
        if 'Income' in test_data.columns:
            income_counts = test_data['Income'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(income_counts.index.astype(str), income_counts.values, color=plt.cm.Set3(np.linspace(0, 1, 2)))
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
