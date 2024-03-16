# Simple-linear-reg
Building a simple linear regression model by performing EDA and all necessary transformations and to select the best model using Python.
##Salary_hike -> Build a prediction model for Salary_hike
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset from CSV file
data = pd.read_csv('Salary_Data.csv')

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Univariate Analysis: Histogram
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.histplot(data['YearsExperience'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Years of Experience')
plt.xlabel('Years of Experience')

plt.subplot(1, 2, 2)
sns.histplot(data['Salary'], bins=10, kde=True, color='salmon')
plt.title('Distribution of Salary')
plt.xlabel('Salary')

plt.tight_layout()
plt.show()

# Bivariate Analysis: Scatter Plot
plt.figure(figsize=(7, 5))
sns.scatterplot(data=data, x='YearsExperience', y='Salary')
plt.title('Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Multivariate Analysis: Pairplot
sns.pairplot(data)
plt.suptitle('Pairplot of Salary Data', y=1.02)
plt.show()

# Linear Regression with Transformational Models
X = data['YearsExperience'].values.reshape(-1, 1)
y = data['Salary']

# Apply transformations
X_log = np.log(X)
X_square = np.square(X)
X_sqrt = np.sqrt(X)

# Transformations dictionary
transformations = {
    'Original': X,
    'Log': X_log,
    'Square': X_square,
    'Sqrt': X_sqrt
}

# Model fitting and evaluation for each transformation
for name, X_transformed in transformations.items():
    print(f"\nTransformation: {name}")
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE:', rmse)

    # Plot the regression line
    plt.figure(figsize=(7, 5))
    plt.scatter(X_test, y_test)
    plt.plot(X_test, y_pred, color='red')
    plt.title(f'Linear Regression ({name}): Salary Prediction')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()
