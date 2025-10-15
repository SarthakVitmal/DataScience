import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('Dataset.csv')
X = data[['Market Share Gaming GPU (%)']]
y = data['Annual Revenue (Billion USD)']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model (for simplicity, we will just print the coefficients here)
import joblib
joblib.dump(model, 'model.pkl')