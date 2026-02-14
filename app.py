"""
Streamlit Web Application for Bank Marketing Term Deposit Prediction
M.Tech (AIML/DSE) - Machine Learning Assignment 2

This app implements the 4 required features per assignment instructions:
1. Dataset upload option (CSV) - 1 mark
2. Model selection dropdown - 1 mark
3. Display of evaluation metrics - 1 mark
4. Confusion matrix - 1 mark
"""

import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
)

# Title
st.title("🏦 Bank Marketing Term Deposit Prediction")
st.markdown("""
This application uses machine learning models trained on the UCI Bank Marketing dataset
to predict whether a client will subscribe to a term deposit.
""")

# Load models
@st.cache_resource
def load_models():
    """Load all trained model pipelines from model/ folder."""
    model_dir = "model"
    models = {}
    
    model_files = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "K-Nearest Neighbors": "k-nearest_neighbors_model.pkl",
        "Naive Bayes": "naive_bayes_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "XGBoost": "xgboost_model.pkl",
    }
    
    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
        else:
            st.warning(f"Model file not found: {path}")
    
    return models

# Load results
@st.cache_data
def load_results():
    """Load model evaluation metrics from CSV."""
    path = os.path.join("model", "model_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Load test data
@st.cache_data
def load_test_data():
    """Load test dataset for confusion matrix."""
    path = os.path.join("model", "test_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Load all data
models = load_models()
results_df = load_results()
test_df = load_test_data()

# FEATURE 2: Model Selection Dropdown (Required - 1 mark)
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("### Select Model")

if models:
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        help="Select from 6 trained classification models"
    )
else:
    st.sidebar.error("No models loaded. Please train models first.")
    selected_model = None

st.sidebar.markdown("---")

# FEATURE 1: Dataset Upload Option (Required - 1 mark)
st.sidebar.markdown("### Upload Test Data (CSV)")

# NEW: Download sample test data from GitHub
st.sidebar.markdown("**Download Sample Test Data:**")
if test_df is not None:
    # Create a sample test dataset (first 100 rows without 'y' column for demonstration)
    sample_test = test_df.drop('y', axis=1).head(100)
    
    csv_sample = sample_test.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Sample CSV (100 rows)",
        data=csv_sample,
        file_name="sample_test_data.csv",
        mime="text/csv",
        help="Download a sample CSV file with 100 test records to try the prediction feature"
    )
    st.sidebar.caption("Use this sample file to test predictions")
else:
    st.sidebar.info("Sample test data not available")

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file without 'y' column",
    type=["csv"],
    help="Upload test data with Bank Marketing features (excluding target column)"
)

if uploaded_file:
    st.sidebar.success("✅ File uploaded!")

# Main content
st.markdown("---")

# Section 1: Model Comparison Table
if results_df is not None:
    st.subheader("📊 Model Comparison")
    st.dataframe(results_df, use_container_width=True)
    st.markdown("---")

# Section 2: FEATURE 3 - Display of Evaluation Metrics (Required - 1 mark)
if selected_model and results_df is not None:
    st.subheader(f"📈 Evaluation Metrics: {selected_model}")
    
    model_row = results_df[results_df['ML Model Name'] == selected_model]
    
    if not model_row.empty:
        row = model_row.iloc[0]
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{row['Accuracy']:.4f}")
        col2.metric("AUC Score", f"{row['AUC']:.4f}" if not pd.isna(row['AUC']) else "N/A")
        col3.metric("Precision", f"{row['Precision']:.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{row['Recall']:.4f}")
        col5.metric("F1 Score", f"{row['F1']:.4f}")
        col6.metric("MCC Score", f"{row['MCC']:.4f}")
        
        st.markdown("---")

# Section 3: FEATURE 4 - Confusion Matrix (Required - 1 mark)
if selected_model and test_df is not None and selected_model in models:
    st.subheader(f"🎯 Confusion Matrix: {selected_model}")
    
    model = models[selected_model]
    
    # Prepare test data
    X_test = test_df.drop('y', axis=1)
    y_true = test_df['y']
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Display as heatmap
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No', 'Predicted: Yes'],
        y=['Actual: No', 'Actual: Yes'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True
    ))
    
    fig_cm.update_layout(
        title=f'Confusion Matrix - {selected_model}',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        height=400,
        width=500
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Display confusion matrix values as table
    st.markdown("**Confusion Matrix Values:**")
    cm_df = pd.DataFrame(
        cm,
        columns=['Predicted: Not Subscribed', 'Predicted: Subscribed'],
        index=['Actual: Not Subscribed', 'Actual: Subscribed']
    )
    st.dataframe(cm_df, use_container_width=True)
    
    st.markdown("---")

# Section 4: Make Predictions on Uploaded Data
if uploaded_file is not None and selected_model in models:
    st.subheader("🔮 Predictions on Uploaded Data")
    
    try:
        # Read uploaded file
        user_df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File loaded successfully! Shape: {user_df.shape}")
        
        # Display preview
        with st.expander("📄 View uploaded data (first 10 rows)"):
            st.dataframe(user_df.head(10), use_container_width=True)
        
        # Prediction button
        if st.button("🚀 Run Predictions", type="primary"):
            with st.spinner(f"Running predictions using {selected_model}..."):
                model = models[selected_model]
                
                # Make predictions
                predictions = model.predict(user_df)
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(user_df)[:, 1]
                else:
                    probabilities = None
                
                # Create results dataframe
                results = user_df.copy()
                results['Prediction'] = ['Subscribed' if p == 1 else 'Not Subscribed' for p in predictions]
                
                if probabilities is not None:
                    results['Probability_Subscribe'] = probabilities
                
                # Display results
                st.success("✅ Predictions completed!")
                st.dataframe(results, use_container_width=True)
                
                # Summary
                subscribed_count = sum(predictions == 1)
                not_subscribed_count = sum(predictions == 0)
                
                col1, col2 = st.columns(2)
                col1.metric("Predicted: Subscribed", subscribed_count)
                col2.metric("Predicted: Not Subscribed", not_subscribed_count)
                
                # Download button
                csv_data = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_data,
                    file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
    
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.info("Please ensure your CSV has the same 20 Bank Marketing columns (without 'y').")

elif uploaded_file is None:
    st.info("👆 Upload a CSV file using the sidebar to make predictions, or download the sample test data above.")

# Section 5: Expected CSV Format
with st.expander("📋 Expected CSV Format for Upload"):
    st.markdown("""
    Your CSV file should contain these **20 columns** (without the target 'y'):
    
    **Client Information:**
    - `age` - Client's age (numeric)
    - `job` - Type of job (admin., technician, blue-collar, services, etc.)
    - `marital` - Marital status (married, single, divorced)
    - `education` - Education level (basic.4y, basic.6y, basic.9y, high.school, university.degree, etc.)
    - `default` - Has credit in default? (yes, no, unknown)
    - `housing` - Has housing loan? (yes, no, unknown)
    - `loan` - Has personal loan? (yes, no, unknown)
    
    **Campaign Contact:**
    - `contact` - Contact type (cellular, telephone)
    - `month` - Last contact month (jan, feb, mar, ..., dec)
    - `day_of_week` - Last contact day (mon, tue, wed, thu, fri)
    - `duration` - Contact duration in seconds (numeric)
    - `campaign` - Number of contacts in this campaign (numeric)
    - `pdays` - Days since last contact from previous campaign (numeric, 999 = not contacted)
    - `previous` - Number of contacts before this campaign (numeric)
    - `poutcome` - Previous campaign outcome (success, failure, nonexistent)
    
    **Economic Indicators:**
    - `emp.var.rate` - Employment variation rate (numeric)
    - `cons.price.idx` - Consumer price index (numeric)
    - `cons.conf.idx` - Consumer confidence index (numeric)
    - `euribor3m` - Euribor 3-month rate (numeric)
    - `nr.employed` - Number of employees (numeric)
    
    **Do NOT include the 'y' column** (target variable) - the model will predict it.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Machine Learning Assignment 2</strong></p>
    <p>M.Tech (AIML/DSE) | BITS Pilani</p>
</div>
""", unsafe_allow_html=True)
