import pandas as pandas
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

filePath = 'iPhone.csv'
data = pd.read_csv(filePath) 

missing_values = data.isnull().sum()
#print(missing_values) # there are no missing values

data_encoded = pd.get_dummies(data, columns=['NAME']) # one-hot encoding. See README.md
y = data_encoded['CO2E'] # extracts dependent variable C02 emissions
X = data_encoded.drop('CO2E', axis=1) # features will be the independent variables. 

# In this case, I will set up a simple linear regression model using all features and perform cross-validation

model = LinearRegression()
k_f = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=k_f, scoring='neg_mean_squared_error')

# I will calculate RMSE to assess the performance and extract the top 5 features based on their importance in the model

rmse_scores = np.sqrt(-cv_scores) # root MSE for each cross-validation fold
model.fit(X,y)
feature_importance = pd.Series(model.coef_, index=X.columns)
top_5_features = feature_importance.abs().nlargest(5)

print("Average RMSE across all cross-validation folds: ", rmse_scores.mean())
# this means that on average, the model's predictions are about 3.45 units away from actual CO2 emissions. 
print("\nTop 5 Important Models: ")
# For example this means one increase in iPhone 12 (holding all other features CONSTANT) = 8.055720 increase in CO2 emissions.
print(top_5_features)



