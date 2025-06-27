import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Create dataset
data = {
    'Area': [1000, 1500, 1800, 2400, 3000],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Price': [200000, 250000, 300000, 350000, 400000]
}
df = pd.DataFrame(data)
print("Dataset:")
print(df)

# Step 2: Feature & Target
X = df[['Area', 'Bedrooms']]
y = df['Price']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)
print("\nPredicted:", y_pred)
print("Actual:", list(y_test))

# Step 6: Evaluation
print("\nEvaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: Visualize
plt.scatter(df['Area'], df['Price'], color='blue')
plt.plot(df['Area'], model.predict(df[['Area', 'Bedrooms']]), color='red')
plt.title('Area vs Price')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# Step 8: Interpret coefficients
print("\nIntercept:", model.intercept_)
print("Coefficients:", model.coef_)
