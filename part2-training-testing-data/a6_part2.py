import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values
x = x.reshape(-1,1)
# Create your training and testing datasets:
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)
# Use reshape to turn the x values into 2D arrays:

# Create the model
model = LinearRegression().fit(xtrain,ytrain)
# Find the coefficient, bias, and r squared values. 
coef = round(float(model.coef_[0]), 2)
intercept = round(float(model.intercept_), 2)
r_squared = model.score(xtrain, ytrain)
# Each should be a float and rounded to two decimal places. 


# Print out the linear equation and r squared value:
print(f"The linear equation is y={coef}x + {intercept}")
print(f"the r^2 value is {r_squared}")


# reshape the xtest data into a 2D array

# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index]
    predicted_y = predict[index]
    x_coord = xtest[index]
    print("x value:", float(x_coord[0]),"Predicted y value:", predicted_y, "Actual y value:", actual)

plt.figure(figsize=(5,4))

#creates a scatter plot and labels the axes
plt.scatter(xtrain,ytrain, c="purple", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing Data")

plt.scatter(xtest, predict, c="red", label="Predictions")

plt.xlabel("Age")
plt.ylabel("blood pressure")
plt.title("Average blood pressure per age group")
plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

plt.legend()
plt.show()