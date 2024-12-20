# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:23:56 2024

@author: Bhumika
"""

#Day 1 : Data Collection and Preparation
import pandas as pd
df = pd.read_csv('C:/Users/Bhumika/Desktop/DA Internship_Personal/Sprint 7/Sprint7_AI_Driven_Survival_Prediction_In_Healthcare/dataset.csv')

# Initialize report_data dictionary
report_data = {}

# Data info
df_info = df.info()
report_data["Data Info"] = str(df_info)

# Summary statistics
summary_stats = df.describe()
report_data["Summary Statistics"] = summary_stats.to_string()

# Visualize target variable distribution
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
ax = sns.countplot(x='hospital_death', data=df)
ax.set_title('Hospital Death Distribution')
ax.set_xlabel('Hospital Death (0 = No Death, 1 = Death)')
ax.set_ylabel('Count')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                fontsize=12, color='black', 
                xytext=(0, 9), textcoords='offset points')
plt.show()

# Save the plot as a file and add to the report data
distribution_plot_path = "hospital_death_distribution.png"
plt.savefig(distribution_plot_path)
report_data["Hospital Death Distribution"] = distribution_plot_path

# Day 2: Data Cleaning and Transformation
#Checking for missing values
print(df.isnull().sum())

# Using imputation for importing mean values for missing values.
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy='mean')  # Impute with the mean for numerical columns
df[df.select_dtypes(include=['float64', 'int64']).columns] = num_imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))

#Checking for missing values
print(df.isnull().sum())

# Imputation for categorical columns
from sklearn.preprocessing import LabelEncoder
cat_imputer = SimpleImputer(strategy='most_frequent')  # Impute with the mode (most frequent value) for categorical columns
df[df.select_dtypes(include=['object']).columns] = cat_imputer.fit_transform(df.select_dtypes(include=['object']))

# Encoding categorical features using LabelEncoder
labelencoder = LabelEncoder()
# Apply LabelEncoder to all categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = labelencoder.fit_transform(df[col])

#Checking for missing values
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

# Summary statistics
summary_stats =cleaned_df.describe()
report_data["Summary Statistics"] = summary_stats.to_string()

#Save cleaned data into SQL
import sqlite3
conn = sqlite3.connect('patient_data.db')
cleaned_df.to_sql('patient_table', conn, if_exists='replace', index=False)

# Day 3 Model Building

#Step 1 Feature Selection

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


#Step 2 Resampling
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
plt.savefig("original_target_distribution.png")  # Save the plot
plt.show()

# After SMOTE - Resampled Distribution
plt.figure(figsize=(10, 5))
sns.countplot(x=y_resampled)
plt.title("Resampled Target Variable Distribution (After SMOTE)")
plt.xlabel("Hospital Death (After SMOTE)")
plt.ylabel("Count")
plt.savefig("resampled_target_distribution.png")
plt.show()

#import 
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
# Model Evaluation and Tuning on resampled data

# Predict the test set results
y_pred_resampled = model_resampled.predict(X_test_resampled_scaled)  
  # Convert probabilities to binary predictions (0 or 1)
y_pred_resampled = (y_pred_resampled > 0.5).astype(int)


# Calculate accuracy, precision, recall, F1-score
accuracy_resampled = accuracy_score(y_test_resampled, y_pred_resampled)
precision_resampled = precision_score(y_test_resampled, y_pred_resampled)
recall_resampled = recall_score(y_test_resampled, y_pred_resampled)
f1_resampled = f1_score(y_test_resampled, y_pred_resampled)

# Print the metrics
print(f"Accuracy on the test set: {accuracy_resampled:.4f}")
print(f"Precision on the test set: {precision_resampled:.4f}")
print(f"Recall on the test set: {recall_resampled:.4f}")
print(f"F1-Score on the test set: {f1_resampled:.4f}")


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

# Save the combined plot as a single image file
plt.savefig("training_history_smote.png")  # Save the plot to a file

plt.show()

#Day 5 Model Interpretation & Deployment
# Use SHAP (SHapley Additive exPlanations) to interpret model predictions.
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
shap.summary_plot(shap_values, X_test_subset, feature_names=X_test_scaled.columns, show=False)

# Step 6:Save the SHAP summary plot
plt.savefig("shap_summary_plot.png", bbox_inches='tight')  # Save the plot to a file

# Show the plot
plt.show()

# Save the Model
model_resampled.save('survival_model.h5')
print("Model saved successfully as 'survival_model.h5'")

#Day 7:Final Visualization & Reporting

# Step 1: Import the Procced data and model data into the csv file
    
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

# Save feature importance to CSV 
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

# Step 2: Generate PDF Report
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from io import StringIO

# Initialize PDF
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Analysis Report', 0, 1, 'C')
        self.ln(10)

    def add_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def add_text(self, text):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, text)
        self.ln(5)

    def add_image(self, image_path, width=190, title=None):
        if title:
            self.add_title(title)
        if os.path.exists(image_path):
            self.image(image_path, x=10, y=None, w=width)
            self.ln(10)
        else:
            self.add_text(f"Image not found: {image_path}")

    def add_table(self, df, title=""):
        self.add_title(title)
        self.set_font('Arial', '', 10)
        col_width = self.w / (len(df.columns) + 1)
        self.set_fill_color(200, 200, 200)

        # Add header
        for column in df.columns:
            self.cell(col_width, 10, column, border=1, align='C', fill=True)
        self.ln()

        # Add rows
        for index, row in df.iterrows():
            for item in row:
                self.cell(col_width, 10, str(item), border=1)
            self.ln()
        self.ln(10)

# Initialize the report
pdf = PDFReport()
pdf.add_page()

# Section 1: Original Dataset Information
pdf.add_title("Original Dataset Information")
data_info_buffer = StringIO()
df.info(buf=data_info_buffer)
data_info_text = data_info_buffer.getvalue()
pdf.add_text(data_info_text)

# Section 2: Cleaned Dataset Information
pdf.add_title("Cleaned Dataset Information")
data_info_buffer = StringIO()
cleaned_df.info(buf=data_info_buffer)
data_info_text = data_info_buffer.getvalue()
pdf.add_text(data_info_text)

# Section 3: List of Selected Features
pdf.add_title("List of Selected Features")
selected_features_list = "\n".join(selected_features)
pdf.add_text(selected_features_list)

# Section 4: Target Variable Distributions
pdf.add_title("Target Variable Distributions")
pdf.add_image("original_target_distribution.png", width=190, title="Original Target Variable Distribution")
pdf.add_image("resampled_target_distribution.png", width=190, title="Resampled Target Variable Distribution")

# Section 5: Accuracy and Loss Plots
pdf.add_title("Training History: Accuracy and Loss Over Epochs")
pdf.add_image("training_history_smote.png", width=190)

# Section 6: Model Metrics
pdf.add_title("Model Metrics")
pdf.add_text(f"Accuracy: {accuracy_resampled:.4f}")
pdf.add_text(f"Precision: {precision_resampled:.4f}")
pdf.add_text(f"Recall: {recall_resampled:.4f}")
pdf.add_text(f"F1-Score: {f1_resampled:.4f}")

# Section 7: SHAP Summary Plot
pdf.add_title("SHAP Summary Plot")
pdf.add_image("shap_summary_plot.png", width=190)

# Save the report
output_path = "analysis_report.pdf"
try:
    pdf.output(output_path)
    print(f"PDF report saved as {output_path}")
except KeyError as e:
    print(f"Error saving the PDF report: {e}")



