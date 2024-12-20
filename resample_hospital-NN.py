# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 20:05:45 2024

@author: steph
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

##### Day 1 - Data Collection and Preparation

# Load the dataset
df = pd.read_csv('C:\\Users\steph\Desktop\IT_Expert_System\Internship\Sprint 7 - Deep Learning\dataset.csv', sep=',')

# Check dataset info
print(df.info())

# Summary statistics
print(df.describe())

# Checking for missing values
print(df.isnull().sum())

# Visualize target variable distribution
sns.countplot(x='hospital_death', data=df)
plt.show()


##### Day 2 - Data Cleaning and Transformation

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Imputation for numerical columns
num_imputer = SimpleImputer(strategy='mean')  # Impute with the mean for numerical columns
df[df.select_dtypes(include=['float64', 'int64']).columns] = num_imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# Imputation for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')  # Impute with the mode (most frequent value) for categorical columns
df[df.select_dtypes(include=['object']).columns] = cat_imputer.fit_transform(df.select_dtypes(include=['object']))

# Encoding categorical features using LabelEncoder
labelencoder = LabelEncoder()
# Apply LabelEncoder to all categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = labelencoder.fit_transform(df[col])

# Check if there are any missing values left
print(df.isnull().sum())

#Checking total rows and columns of table
rows, columns = df.shape
print("\nTotal rows:", {rows})
print("Total columns:", {columns})

# Identify columns with negative numbers
numer_columns = df.select_dtypes(include=['float64', 'int64']).columns
columns_with_negatives = [col for col in numer_columns if (df[col] < 0).any()]
# Print the results
print("\nColumns with Negative Numbers:")
print(columns_with_negatives)

# Remove extreme outliers using IQR method for specified columns only
def remove_extreme_outliers(df, specific_columns, multiplier=2.5):
    cleaned_df = df.copy()  # Create a copy of the original dataframe
    outliers_dict = {}  # To store outliers for each column
    
    for col in specific_columns:
        # Calculate Q1 (10th percentile) and Q3 (90th percentile)
        Q1 = df[col].quantile(0.07)  # Lower bound
        Q3 = df[col].quantile(0.93)  # Upper bound
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define extreme outlier bounds
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Find outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_dict[col] = outliers
        
        # Print outliers for the column
        if not outliers.empty:
            print("\nOutliers in column" ,{col},":")
            print(outliers[[col]])  # Print only the outlier values for the column
            
        # Remove rows with extreme outliers
        cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound)]
    
    return cleaned_df, outliers_dict

specific_columns = ['apache_4a_icu_death_prob','apache_4a_hospital_death_prob','pre_icu_los_days']  # Specify the three columns to check for outliers
cleaned_df, outliers_dict = remove_extreme_outliers(df, specific_columns, multiplier=2.5)

# Identify columns with negative numbers
num_columns = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
columns_negatives = [col for col in num_columns if (cleaned_df[col] < 0).any()]
# Print the results
print("\nColumns with Negative Numbers:")
print(columns_negatives)

cleaned_df = cleaned_df[cleaned_df['pre_icu_los_days'] >= 0]

num_columns = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
columns_negatives = [col for col in num_columns if (cleaned_df[col] < 0).any()]
# Print the results
print("\nColumns with Negative Numbers after removal of negative num:")
print(columns_negatives)

# Print cleaned data
print("\nCLEANED HEALTHCARE DATA:")
print(cleaned_df)

#Save cleaned data into SQL
import sqlite3
conn = sqlite3.connect('patient_data.db')
cleaned_df.to_sql('patient_table', conn, if_exists='replace', index=False)

query = "SELECT * FROM patient_table LIMIT 5;"  # Get first 5 rows to check the data
df_from_sql = pd.read_sql_query(query, conn)
print(df_from_sql)

##### Day 3 - Model Building

# Feature Selection

from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Split data into features and target
X = cleaned_df.drop(columns=['hospital_death', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob'])
y = cleaned_df['hospital_death']

# Select only numeric features
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
X_numeric = X[numeric_columns]

# Use SelectKBest with mutual_info_classif to select top numeric features
selector = SelectKBest(score_func=mutual_info_classif, k=10)  # Adjust k to select top 'k' numeric features
X_new = selector.fit_transform(X_numeric, y)

# Get the selected numeric feature names
selected_features = X_numeric.columns[selector.get_support()]

# Display the selected numeric features
print("Selected Numeric Features based on Mutual Information:")
print(selected_features)


# Create a DataFrame with selected features and target
X = cleaned_df[selected_features]
y = cleaned_df['hospital_death']

# Resampling
from imblearn.over_sampling import SMOTE


# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)

# Apply SMOTE on the selected features and target variable
X_resampled, y_resampled = smote.fit_resample(X, y)

# Before SMOTE - Original Distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=y)
plt.title("Original Target Variable Distribution")
plt.xlabel("Hospital Death (Before SMOTE)")
plt.ylabel("Count")
plt.show()

# After SMOTE - Resampled Distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=y_resampled)
plt.title("Resampled Target Variable Distribution (After SMOTE)")
plt.xlabel("Hospital Death (After SMOTE)")
plt.ylabel("Count")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Split the resampled data into train/test datasets
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Data Standardization
scaler_resampled = StandardScaler()
X_train_resampled_scaled = scaler_resampled.fit_transform(X_train_resampled)
X_test_resampled_scaled = scaler_resampled.transform(X_test_resampled)

# Build Keras Model
model_resampled = Sequential()
model_resampled.add(Dense(32, input_dim=X_train_resampled_scaled.shape[1], activation='relu'))
model_resampled.add(Dense(16, activation='relu'))
model_resampled.add(Dense(1, activation='sigmoid'))

# Compile the model
model_resampled.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the resampled data
history_resampled = model_resampled.fit(X_train_resampled_scaled, y_train_resampled, epochs=10, batch_size=32,
                                         validation_data=(X_test_resampled_scaled, y_test_resampled))


# Day 4 - Model Evaluation and Tuning (on resampled data)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

y_pred_resampled = model_resampled.predict(X_test_resampled_scaled)  # Predict the test set results
y_pred_resampled = (y_pred_resampled > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

# Calculate accuracy, precision, recall, F1-score
accuracy_resampled = accuracy_score(y_test_resampled, y_pred_resampled)
precision_resampled = precision_score(y_test_resampled, y_pred_resampled)
recall_resampled = recall_score(y_test_resampled, y_pred_resampled)
f1_resampled = f1_score(y_test_resampled, y_pred_resampled)

# Print evaluation metrics
print(f"Test Set Evaluation Metrics (After SMOTE):")
print(f"Accuracy: {accuracy_resampled*100:.2f}%")
print(f"Precision: {precision_resampled:.2f}")
print(f"Recall: {recall_resampled:.2f}")
print(f"F1-Score: {f1_resampled:.2f}")

# Plot training history (Accuracy and Loss curves over epochs)
# Accuracy and loss data from the training history
train_loss_resampled = history_resampled.history.get('loss', [])
val_loss_resampled = history_resampled.history.get('val_loss', [])
train_accuracy_resampled = history_resampled.history.get('accuracy', [])
val_accuracy_resampled = history_resampled.history.get('val_accuracy', [])

# Plot accuracy curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy_resampled, label='Train Accuracy')
plt.plot(val_accuracy_resampled, label='Validation Accuracy')
plt.title('Accuracy over Epochs (SMOTE Applied)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss curves
plt.subplot(1, 2, 2)
plt.plot(train_loss_resampled, label='Train Loss')
plt.plot(val_loss_resampled, label='Validation Loss')
plt.title('Loss over Epochs (SMOTE Applied)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


##### Day 5 - Model Interpretation and Deployment

# Saved model####################################
model_resampled.save('smote_nn_model.h5')
from keras.models import load_model
# Load the model
loaded_model = load_model('smote_nn_model.h5')

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Generate predictions with the loaded model
y_pred_loaded = loaded_model.predict(X_test_resampled_scaled)
y_pred_binary = (y_pred_loaded > 0.5).astype(int)  # Convert probabilities to binary predictions (0 or 1)

# Calculate metrics
accuracy_loaded = accuracy_score(y_test_resampled, y_pred_binary)
precision_loaded = precision_score(y_test_resampled, y_pred_binary)
recall_loaded = recall_score(y_test_resampled, y_pred_binary)
f1_loaded = f1_score(y_test_resampled, y_pred_binary)

# Print the metrics
print(f"Loaded Model Test Accuracy: {accuracy_loaded:.2f}")
print(f"Loaded Model Test Precision: {precision_loaded:.2f}")
print(f"Loaded Model Test Recall: {recall_loaded:.2f}")
print(f"Loaded Model Test F1-Score: {f1_loaded:.2f}")
##########################################################


import shap
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Step 1: Use the existing resampled and standardized data
X_train_scaled = pd.DataFrame(X_train_resampled_scaled, columns=X_resampled.columns)
X_test_scaled = pd.DataFrame(X_test_resampled_scaled, columns=X_resampled.columns)

# Step 2: Convert the Keras model to SHAP-compatible prediction function
def model_predict(data):
    return model_resampled.predict(data, verbose=0).flatten()

# Step 3: Initialize SHAP Explainer with KernelExplainer for non-tree models
background_data = shap.kmeans(X_train_scaled, 10)  # Summarize background for KernelExplainer
explainer = shap.KernelExplainer(model_predict, background_data)

# Step 4: Compute SHAP values for a subset of the test data
X_test_subset = X_test_scaled[:50]  # Select 50 samples to speed up computation
shap_values = explainer.shap_values(X_test_subset)

# Step 5: Visualize SHAP summary plot
shap.summary_plot(shap_values, X_test_subset, feature_names=X_test_scaled.columns)


#Day 7:Final Visualization & Reporting

# Step 1: 
# Create a DataFrame with only the test data and add predictions
df_resampleddata_modeldata = pd.DataFrame(X_test_resampled, columns=selected_features)  # Include only the test features
df_resampleddata_modeldata['hospital_death'] = y_test_resampled  # Add the target for test set
df_resampleddata_modeldata['predicted_survival_probability'] = y_pred_resampled  # Add predicted probabilities
df_resampleddata_modeldata['predicted_survival_label'] = y_pred_resampled  # Add binary labels

# Export the test data with predictions to CSV (or Excel) for Power BI
df_resampleddata_modeldata.to_csv('processed_test_data_with_predictions.csv', index=False)

print("Processed test data with predictions and binary labels exported successfully.")

# Print the shape of the DataFrame (number of rows and columns)
print(f"Shape of the DataFrame: {df_resampleddata_modeldata.shape}")

# Print the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:")
print(df_resampleddata_modeldata.head())

# Print the number of missing values in each column
print("\nMissing Values in each Column:")
print(df_resampleddata_modeldata.isnull().sum())


# Import the importance of features into csv file

import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed  # For parallel processing

# Custom scoring function for permutation importance
def custom_accuracy_score(estimator, X, y):
    # Predict probabilities
    y_pred = estimator.predict(X)
    # Convert probabilities to binary predictions (0 or 1)
    y_pred_binary = (y_pred > 0.5).astype(int)
    # Calculate accuracy score
    return accuracy_score(y, y_pred_binary)

# Predict the test set results using your original code
y_pred_resampled = model_resampled.predict(X_test_resampled_scaled)  
# Convert probabilities to binary predictions (0 or 1)
y_pred_resampled = (y_pred_resampled > 0.5).astype(int)

# Function to calculate permutation importance for a single feature
def calculate_feature_importance(X, y, estimator, feature_idx):
    X_temp = X.copy()
    X_temp[:, feature_idx] = np.random.permutation(X_temp[:, feature_idx])
    return custom_accuracy_score(estimator, X_temp, y)

# Perform permutation importance with parallelization
import numpy as np

n_features = X_test_resampled_scaled.shape[1]
importances = []

# Run permutation importance in parallel to speed it up
for feature_idx in range(n_features):
    permuted_scores = Parallel(n_jobs=-1)(delayed(calculate_feature_importance)(
        X_test_resampled_scaled, y_test_resampled, model_resampled, feature_idx) 
        for _ in range(5))  # Reduce n_repeats here to speed up
    importances.append(np.mean(permuted_scores) - custom_accuracy_score(model_resampled, X_test_resampled_scaled, y_test_resampled))

# Create a DataFrame with the feature names and their corresponding importance scores
feature_importance_df = pd.DataFrame({
    'Feature': selected_features,  # Use the selected features from your model
    'Importance': importances  # Average importance score across the repetitions
})

# Save feature importance to CSV (or Excel) for Power BI
feature_importance_df.to_csv('feature_importance.csv', index=False)

print("Feature importance exported successfully.")

# Print the shape of the DataFrame (number of rows and columns)
print(f"Shape of the DataFrame: {df_resampleddata_modeldata.shape}")

# Print the first few rows of the feature importance DataFrame
print("\nFeature Importance (Top 5 features):")
print(feature_importance_df.head())

# Print the number of missing values in each column
print("\nMissing Values in each Column:")
print(df_resampleddata_modeldata.isnull().sum())

#Step 2 Generate PDF


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_pdf():
    c = canvas.Canvas("Model_Report.pdf", pagesize=letter)

    # Add title and accuracy to the PDF
    c.drawString(100, 750, "Patient Survival Prediction Model Report")
    c.drawString(100, 730, f"Accuracy: {accuracy_resampled*100:.2f}%")

    # Add selected features to the report
    c.drawString(100, 710, "Selected Features based on Mutual Information:")
    y_offset = 690
    for feature in selected_features:
        c.drawString(100, y_offset, feature)
        y_offset -= 20  # Move down the page for each feature

    c.save()
    # Print confirmation message after saving the PDF
    print("PDF saved successfully as 'Model_Report.pdf'")

# Call the function to create the PDF
create_pdf()








