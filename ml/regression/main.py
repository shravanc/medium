# Step 1
# Load the Data

# Step 2
# Process the Data

# Step 3
# Divide the data into test and train data

# Step 4
# Import the required classificiation library
# LinearRegression

# Step 5
# Analyse the result with Accuracy for any tuning

#****************************************************

#Step 1
import pandas as pd
data = pd.read_csv("weather_lab.csv")
print(data.head)


#Step 2
X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.xlabel('MinTemp', fontsize=14)
plt.ylabel('MaxTemp', fontsize=14)
plt.show()

X = X.reshape(-1, 1)


#Step 3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


#Step 4
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train, y_train)
#print(x_test)
y_pred_test   = clf.predict(x_test)
y_pred_train  = clf.predict(x_train)


#Step5
# Least Mean Sqaured error calculations.
import numpy as np
print(np.sqrt(np.mean((y_pred_test - y_test)**2)))

alpha   = clf.intercept_
beta    = clf.coef_[0]
plt.plot(X,alpha+beta*X,color='r')
plt.scatter(X,y)
plt.show()


#Step 6
minimum_temp = input("Enter Minimum Temp:")
print(clf.predict(np.array([[int(minimum_temp)]])))

