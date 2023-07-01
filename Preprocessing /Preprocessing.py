from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = df.drop('customerID', axis=1)

# Handle missing values and convert to float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing values with 0
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Convert to float
df['TotalCharges'] = df['TotalCharges'].astype(float)

df = df.fillna(df.median(numeric_only=True))

# Encode categorical variables
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']
le = LabelEncoder()
df[cat_cols] = df[cat_cols].apply(lambda x: le.fit_transform(x))

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
