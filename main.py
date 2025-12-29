# ===============================
# House Price Prediction Project
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
import joblib


# Load Data
df = pd.read_csv('Self_Learning_Through_Projects/House_Price_Prediction/data/house_data.csv')
# print(df.head)


# Data Understanding
# print(df.info())
# print(df.describe())

# Check Missing Values
# print(df.isnull().sum())

# EDA
# sns.scatterplot(x='area',y='price',data=df)
# plt.title("Area Vs Price")
# plt.show()

# plt.figure(figsize=(6,4))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()

# 5. Feature Selection
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']


# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )


# 7. Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# 8. Evaluate Model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 9. Prediction
prediction = model.predict([[1400, 3, 2]])
print("Predicted Price:", prediction)

# 10. Save Model
joblib.dump(model, '/Users/nikhilkumar/Desktop/Ds/Self_Learning_Through_Projects/House_Price_Prediction/model/house_price_model.pkl')
print("Model saved successfully!")