import streamlit as st
import json
import os
import pickle
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

# Load data from JSON files
@st.cache_data
def load_results():
    if os.path.exists('model/model_results.json'):
        with open('model/model_results.json', 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_test_data():
    if os.path.exists('model/test_data.json'):
        with open('model/test_data.json', 'r') as f:
            return json.load(f)
    return None

@st.cache_resource
def load_income_mapping():
    if os.path.exists('model/income_mapping.pkl'):
        with open('model/income_mapping.pkl', 'rb') as f:
            return pickle.load(f)
    return {0: "≤ $50K", 1: "> $50K"}

# Title
st.title("💰 Adult Income Classification")
st.markdown("### ML Assignment: Binary Classification with Multiple Models")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["📊 Model Metrics", "🧪 Make Predictions", "📈 Model Comparison", "📥 Dataset Info"]
)

# Load resources
income_mapping = load_income_mapping()
results_data = load_results()
test_data = load_test_data()

if page == "📊 Model Metrics":
    st.header("Model Performance Metrics")
    
    if results_data:
        st.subheader("Evaluation Metrics Comparison")
        st.dataframe(results_data, use_container_width=True)
        
        # Download
        json_str = json.dumps(results_data, indent=2)
        st.download_button(
            label="📥 Download Metrics (JSON)",
            data=json_str,
            file_name="model_metrics.json",
            mime="application/json"
        )
        
        # Best models
        st.subheader("🏆 Best Models by Metric")
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        best_acc = max(results_data, key=lambda x: x.get('Accuracy', 0))
        best_f1 = max(results_data, key=lambda x: x.get('F1', 0))
        best_auc = max(results_data, key=lambda x: x.get('AUC', 0))
        best_prec = max(results_data, key=lambda x: x.get('Precision', 0))
        best_rec = max(results_data, key=lambda x: x.get('Recall', 0))
        best_mcc = max(results_data, key=lambda x: x.get('MCC', 0))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            model_key = [k for k in best_acc.keys() if k not in metrics][0]
            st.metric("Best Accuracy", best_acc[model_key], f"{best_acc.get('Accuracy', 0):.4f}")
            st.metric("Best F1 Score", best_f1.get(model_key, 'N/A'), f"{best_f1.get('F1', 0):.4f}")
        with col2:
            st.metric("Best AUC", best_auc.get(model_key, 'N/A'), f"{best_auc.get('AUC', 0):.4f}")
            st.metric("Best Precision", best_prec.get(model_key, 'N/A'), f"{best_prec.get('Precision', 0):.4f}")
        with col3:
            st.metric("Best Recall", best_rec.get(model_key, 'N/A'), f"{best_rec.get('Recall', 0):.4f}")
            st.metric("Best MCC", best_mcc.get(model_key, 'N/A'), f"{best_mcc.get('MCC', 0):.4f}")
        
        # Visualization
        st.subheader("📉 Metrics Visualization")
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Model Performance Across Metrics', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        model_names = [k for k in results_data[0].keys() if k not in metrics]
        for idx, metric in enumerate(metrics):
            values = [row.get(metric, 0) for row in results_data]
            axes[idx].barh(model_names, values, color=plt.cm.Set3(range(len(results_data))))
            axes[idx].set_xlabel(metric)
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_xlim(0, 1)
            for i, v in enumerate(values):
                axes[idx].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)

elif page == "🧪 Make Predictions":
    st.header("Test Model Predictions")
    st.info("ℹ️ **Static Results (Pre-Computed)**\n\nModels have been trained locally. To test custom predictions, download the test data and run the models locally using the repository code.")

elif page == "📈 Model Comparison":
    st.header("Model Performance Comparison")
    
    if results_data:
        st.subheader("Model Performance Overview")
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        model_names = [k for k in results_data[0].keys() if k not in metrics]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Model Performance Across Metrics', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            values = [row.get(metric, 0) for row in results_data]
            axes[idx].barh(model_names, values, color=plt.cm.Set3(range(len(results_data))))
            axes[idx].set_xlabel(metric)
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_xlim(0, 1)
            for i, v in enumerate(values):
                axes[idx].text(v + 0.02, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Heatmap
        st.subheader("Heatmap - All Metrics")
        fig, ax = plt.subplots(figsize=(10, 6))
        heatmap_values = [[row.get(m, 0) for m in metrics] for row in results_data]
        sns.heatmap(heatmap_values, annot=True, fmt='.4f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Score'}, ax=ax, vmin=0, vmax=1,
                   xticklabels=metrics, yticklabels=model_names)
        ax.set_title('Model Performance Heatmap', fontweight='bold', fontsize=12)
        st.pyplot(fig)

elif page == "📥 Dataset Info":
    st.header("Dataset Information")
    st.subheader("Adult Income Census Dataset")
    st.markdown("""
    **Dataset Overview:**
    - **Source:** UCI Machine Learning Repository
    - **Target:** Binary Income Classification (≤ $50K, > $50K)
    - **Instances:** 32,561 samples | **Features:** 14 attributes
    
    **Description:**
    The Adult dataset is extracted from the 1994 U.S. Census database. It is commonly used 
    for binary classification tasks to predict whether an individual's income exceeds $50K/year 
    based on demographic information.
    """)
    
    if test_data:
        st.subheader("Test Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", len(test_data))
        with col2:
            st.metric("Features", len(test_data[0]) if test_data else 0)
        with col3:
            income_vals = set(row.get('Income', '') for row in test_data)
            st.metric("Classes", len(income_vals))
        
        st.subheader("Sample Data")
        st.dataframe(test_data[:10], use_container_width=True)
        
        st.subheader("Income Distribution")
        income_counts = {}
        for row in test_data:
            income = row.get('Income', 'Unknown')
            income_counts[income] = income_counts.get(income, 0) + 1
        
        if income_counts:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(income_counts.keys(), income_counts.values(), color=plt.cm.Set3(range(len(income_counts))))
            ax.set_ylabel('Count')
            ax.set_title('Income Distribution')
            ax.tick_params(axis='x', rotation=45)
            for i, (k, v) in enumerate(income_counts.items()):
                ax.text(i, v + 100, str(v), ha='center')
            plt.tight_layout()
            st.pyplot(fig)
        
        st.subheader("📥 Download Test Data")
        json_str = json.dumps(test_data, indent=2)
        st.download_button(
            label="Download Test Data (JSON)",
            data=json_str,
            file_name="test_data.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><b>ML Classification Assignment - Adult Income Prediction</b></p>
    <p>Built with Streamlit | Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost</p>
</div>
""", unsafe_allow_html=True)
