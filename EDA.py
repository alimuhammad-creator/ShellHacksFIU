# EDA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    df = pd.read_csv("Transactional_dataset.csv")
    return df

def plot_transaction_type(df):
    sns.set_style("dark")
    sns.set_palette("pastel")
    plt.figure(figsize=(8, 6))
    df['type'].value_counts().plot(kind='bar', color='#808080')
    plt.title('Type of transaction', color='#000000', fontsize=20)
    plt.xticks(rotation=45, color='#000000')
    plt.xlabel('Type', fontsize=18, color='#000000')
    plt.ylabel('Count', fontsize=18, color='#000000')
    plt.tight_layout()
    plt.show()

def plot_transaction_amount(df):
    sns.set_style("dark")
    sns.set_palette("pastel")
    plt.figure(figsize=(10, 5))
    df['amount'].value_counts().sort_values(ascending=False).head().plot(kind='bar', color='#808080')
    plt.title("Amount of the transaction", fontsize=20, color="#000000")
    plt.xticks(rotation=0, fontsize=12, color='#000000')
    plt.xlabel('Amount', fontsize=16, color='#000000')
    plt.ylabel('Count', fontsize=16, color='#000000')
    plt.tight_layout()
    plt.show()

def plot_transaction_pie_chart(df):
    counts = df.groupby('type').count()['amount']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#CCB974']

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=colors, shadow=True,
            explode=(0.1, 0, 0, 0, 0), textprops={'fontsize': 15})
    plt.title('Count of each type of transaction', fontweight='bold', fontsize=18, fontfamily='times new roman')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns
    numeric_data = df[numeric_columns]
    correlation = numeric_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, vmin=-1, vmax=1, cmap="Greys", annot=True, fmt='.2f')
    plt.title('Pearson Correlation Matrix', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_all_plots():
    df = load_and_preprocess_data()
    plot_transaction_type(df)
    plot_transaction_amount(df)
    plot_transaction_pie_chart(df)
    plot_correlation_matrix(df)
