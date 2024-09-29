# streamlit_app.py
import streamlit as st
import pandas as pd
from model import preprocess_data, train_model, evaluate_model, predict_new_data, load_and_train_model
from EDA import generate_all_plots, display_top_20_fraudulent_transactions, load_and_preprocess_data  

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Sidebar 
st.sidebar.title("Upload Transaction Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

# Button to get insights
if st.sidebar.button('Get Insights'):
    st.write("### Getting Insights from Transactions....")
    generate_all_plots() 

    # EDA graphs
    st.image('transaction_type.png', caption='Transaction Type Distribution')
    st.image('transaction_amount.png', caption='Top Transaction Amounts')
    st.image('transaction_pie_chart.png', caption='Count of Each Type of Transaction')
    st.image('correlation_matrix.png', caption='Pearson Correlation Matrix')
    st.image('top_5_users_biggest_transactions.png', caption='Top 5 Users with Biggest Transactions')
    st.image('top_5_users_most_transactions.png', caption='Top 5 Users with Most Transactions')

# New button to view top 20 fraudulent transactions
if st.sidebar.button('View Fraudulent Transactions'):
    st.write("### Top 20 Fraudulent Transactions")
    df = load_and_preprocess_data()  # Load the dataset
    top_20_fraudulent = display_top_20_fraudulent_transactions(df)  # Get top 20 fraudulent transactions
    st.dataframe(top_20_fraudulent)  # Display the table

# Load and process dataset for prediction
if uploaded_file is not None:
    # Load and train model on the original dataset
    model, le, scaler = load_and_train_model("Transactional_dataset.csv", "Decision Tree")

    # Upload new data and make predictions
    st.write("### Predict on New Data")
    new_file = st.file_uploader("Upload a CSV for Predictions", type=["csv"], key="new_data")
    if new_file:
        custom_df = pd.read_csv(new_file)
        # Preprocess new data
        custom_df = preprocess_data(custom_df, fit_scaler=False)  # Use scaler and label encoder trained on original dataset
        predictions = predict_new_data(model, custom_df, le, scaler)
        custom_df['Predicted_Fraud'] = predictions
        
        # Display only fraudulent transactions
        fraudulent_transactions = custom_df[custom_df['Predicted_Fraud'] == 1]
        st.write("### Fraudulent Transactions")
        st.dataframe(fraudulent_transactions[['type', 'amount', 'Predicted_Fraud']])
