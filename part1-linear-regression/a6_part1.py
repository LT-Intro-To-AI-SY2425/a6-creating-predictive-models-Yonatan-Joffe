import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1,1)

# Create the model
model = LinearRegression().fit(x,y)
# Find the coefficient, bias, and r squared values. 
coef = round(float(model.coef_[0]), 2 )
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)
# Print out the linear equation and r squared value
print(f"The linear equation is y = {coef}x + {intercept}")
print(f"The r^2 value is {r_squared}")
# Predict the the blood pressure of someone who is 43 years old.
x_prediction = 42
prediction = model.predict([[x_prediction]])
print(f"the predicted blood pressure for a {x_prediction} year old is {prediction}")

# Create the model in matplotlib and include the line of best fit
plt.figure(figsize = (6,4))

plt.scatter(x,y,c = "orange")
plt.scatter(x_prediction,prediction,c = "blue")
plt.xlabel("age")
plt.ylabel("blood pressure")
plt.title("average blood pressure per age group")
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")
plt.show()
plt.legend()