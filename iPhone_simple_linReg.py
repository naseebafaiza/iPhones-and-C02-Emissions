from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

filePath = 'iPhone.csv'
data = pd.read_csv(filePath) 

# one-hot encoding on the 'NAME' column, see README.md
data_encoded = pd.get_dummies(data, columns=['NAME'])
y = data_encoded['CO2E']  # Dependent variable

# Initializing dictionary to store the coefficient for each feature
feature_coefficient = {}

for feature in data.columns:
    if feature not in ['CO2E', 'NAME']:

        X_feature = data_encoded[[col for col in data_encoded.columns if col.startswith(feature) or col == feature]]
        model = LinearRegression()
        model.fit(X_feature, y)
        
        # This coefficient represents the change in CO2E for a 1 unit increase in the feature
        feature_coefficient[feature] = model.coef_[0]

# Print the coefficient for each feature
for feature, coef in feature_coefficient.items():
    print(f"{feature}: {coef:.4f} increase in CO2E per unit increase")
