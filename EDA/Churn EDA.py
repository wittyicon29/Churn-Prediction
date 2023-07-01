import pandas as pd

df = pd.read_csv('Telco-Customer-Churn.csv')

df.head().T

print(f"Dataset Dimensions : {df.shape}")

print(df.dtypes)

df.isnull().sum()

df.describe().T

categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

for col in categorical_cols:
    print(f"Unique values in {col}: {df[col].unique()}")

import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Visualize the churn rate based on various categorical features
for feature in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='Churn', data=df)
    plt.title(f'Churn Rate by {feature}')
    plt.xticks(rotation=45)
    plt.show()

# Convert 'Churn' column to numeric format
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Visualize the churn rate based on numerical features
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

for feature in num_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y=feature, data=df)
    plt.title(f'{feature} Distribution by Churn')
    plt.show()

# Visualize the distribution of numerical features
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

for feature in num_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

# Visualize the distribution of categorical columns using pie charts
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

for feature in categorical_cols:
    plt.figure(figsize=(8, 6))
    df[feature].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title(f'{feature} Distribution')
    plt.ylabel('')
    plt.show()

# Visualize the distribution of categorical columns using bar plots
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

for feature in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, data=df, hue='Churn')
    plt.title(f'{feature} Distribution by Churn')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Visualize the churn distribution based on numerical features using box plots
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

for feature in num_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y=feature, data=df)
    plt.title(f'{feature} Distribution by Churn')
    plt.xlabel('Churn')
    plt.ylabel(feature)
    plt.show()

# Visualize the stacked bar plot of churned vs non-churned customers based on categorical columns
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

for feature in categorical_cols:
    plt.figure(figsize=(10, 6))
    churn_counts = df.groupby([feature, 'Churn']).size().unstack()
    churn_counts.plot(kind='bar', stacked=True)
    plt.title(f'Stacked Bar Plot of Churned vs Non-Churned Customers by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(['No Churn', 'Churn'])
    plt.show()

# Visualize the churn distribution based on numerical features using violin plots
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

for feature in num_features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Churn', y=feature, data=df)
    plt.title(f'{feature} Distribution by Churn')
    plt.xlabel('Churn')
    plt.ylabel(feature)
    plt.show()


# Visualize the pairwise relationships between select numerical features
selected_num_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']

sns.pairplot(df[selected_num_features], hue='Churn', corner=True)
plt.title('Pairwise Relationships between Numerical Features')
plt.show()

# Compute the correlation matrix
corr_matrix = df.corr()

# Generate a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
