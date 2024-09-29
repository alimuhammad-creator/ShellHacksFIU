import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report

# Load the dataset
df = pd.read_csv("Transactional_dataset.csv")

# Print the number of rows and columns
print(df.count())

# Function to preprocess data
def preprocess_data(df):
    # Remove unnecessary columns
    df.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'nameOrig', 'nameDest'], axis=1, inplace=True)

    # Encoding categorical columns
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    # Separating feature variables and class variable
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    # Standardizing the data
    sc = StandardScaler()
    X = sc.fit_transform(X)

    return X, y, le, sc

# Function to train the model
def train_model(model_name, X_train, y_train):
    if model_name == 'Logistic Regression':
        model = LogisticRegression(class_weight='balanced')  # Assign weights
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=20, class_weight='balanced')  # Assign weights
    elif model_name == 'MLP Classifier':
        model = MLPClassifier(hidden_layer_sizes=(10,), batch_size=32, learning_rate='adaptive', learning_rate_init=0.001)
    else:
        raise ValueError("Model not recognized.")

    model.fit(X_train, y_train)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, precision, recall, report

# Function to make predictions on new data
def predict_new_data(model, custom_df, le, scaler):
    custom_df['type'] = le.transform(custom_df['type'])
    custom_df = scaler.transform(custom_df)
    predictions = model.predict(custom_df)
    return predictions

# Preprocess the data
X, y, le, scaler = preprocess_data(df)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train and evaluate models
def train_evaluate_model(model, model_name):
    # Fitting the model
    model.fit(X_train, y_train)
    
    # Predicting the test set results
    y_pred = model.predict(X_test)
    
    # Evaluating performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    # Printing performance metrics
    print(f"Accuracy of {model_name}: {accuracy:.4f}")
    print(f"Precision of {model_name}: {precision:.4f}")
    print(f"Recall of {model_name}: {recall:.4f}")
    print(f"Classification Report of {model_name}:\n{classification_rep}")
    
    return accuracy, precision, recall

# Function to load and train the model on the original dataset
def load_and_train_model(dataset_path, model_name):
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Preprocess the data
    X, y, le, scaler = preprocess_data(df)

    # Train the model
    model = train_model(model_name, X, y)

    return model, le, scaler

# Train and evaluate each model
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, class_weight='balanced'),
    #"MLP Classifier": MLPClassifier(hidden_layer_sizes=(10,), batch_size=32, learning_rate='adaptive', learning_rate_init=0.001)
}

performance_metrics = []

for model_name, model in models.items():
    accuracy, precision, recall = train_evaluate_model(model, model_name)
    performance_metrics.append((model_name, accuracy, precision, recall))

# Compare Models
performance_df = pd.DataFrame(performance_metrics, columns=['Model', 'Accuracy', 'Precision', 'Recall'])

print("\nModel Comparison:")
print(performance_df)

# Visualizing Performance Comparison
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

metrics = ['Accuracy', 'Precision', 'Recall']

for i, metric in enumerate(metrics):
    performance_df.plot(kind='bar', x='Model', y=metric, ax=ax[i], color='#808080')
    ax[i].set_xlabel('Models', fontsize=14)
    ax[i].set_ylabel(metric, fontsize=14)
    ax[i].set_title(f'{metric} by Model', fontsize=20, fontweight='bold')
    ax[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
