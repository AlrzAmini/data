import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load data
file_path = 'dataset.xlsx'
with pd.ExcelFile(file_path) as xls:
    df = pd.read_excel(xls)

# unneccesary
df = df.drop(columns=['type', 'make'])

# Check for and handle missing values
df.fillna(method='ffill', inplace=True)  # Example method, choose appropriate handling

# Feature Engineering
df['sal_karkard_ratio'] = df['karkard'] / df['sal']
df['tamiri_motor'] = (df['motor'] == 'tamiri').astype(int)
df['dogane_sookht'] = (df['sookht'] == 'dogane').astype(int)


# Define columns for one-hot encoding and scaling
categorical_features = ['subtype', 'sookht', 'rang', 'motor', 'shasi_jolo', 'shasi_aghab', 'badane']
numeric_features = ['sal', 'karkard', 'tamiri_motor', 'sal_karkard_ratio','dogane_sookht']

# Separate features and target
X = df.drop('gheymat', axis=1)
y = df['gheymat']


# Preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'  # Keeps the remaining columns
)

# Create pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),  # Scaler
    ('regressor', LinearRegression())
])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
r_squared = pipeline.score(X_test, y_test)
print(f'R-squared score: {r_squared:.2f}')

# Predict and save results
y_pred = pipeline.predict(X_test)
df['predicted_gheymat'] = pipeline.predict(X)

# Save to excel
output_file_path = 'result_predictions3.xlsx'
df.to_excel(output_file_path, index=False)

print(df.head())
print(df.info())
