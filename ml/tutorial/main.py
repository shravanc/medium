# Step 1
# Load the Data

# Step 2
# Process the Data

# Step 3
# Divide the data into test and train data

# Step 4
# Import the required classificiation library
# MultinomiaNB

# Step 5
# Analyse the result with Accuracy for any tuning

# Step 6
# Predict the new data


#Step 1
import pandas as pd
veg  = pd.read_csv("vegetables.csv")
frt  = pd.read_csv("fruits.csv")

vegetables = [(v[0], "vegetable") for v in veg.values]
fruits     = [(f[0], "fruit") for f in frt.values]
items      = vegetables + fruits

import random
random.shuffle(items)



#Step 2
import numpy as np
items = np.array(items)

X = [item[-1] for item in items[:, 0]]
y = np.where(items[:, 1] == "vegetable",0,1)

# labellising the last letter
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(X)
X2 = lb.transform(X)

#Step 3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.4, random_state=42)


#Step 4
from sklearn.naive_bayes import MultinomialNB

#y = alpha + beta * X
clf = MultinomialNB(alpha=0.1, fit_prior=True)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train) 
y_test_pred  = clf.predict(X_test)

#Step 5
from sklearn.metrics import accuracy_score, confusion_matrix

print(accuracy_score(y_train_pred, y_train))
print(accuracy_score(y_test_pred, y_test))

CC=confusion_matrix(y_test,y_test_pred)
print(CC)


#Step 6
# Client
predict_item = input("Enter your name : ") 
#predict_item = "mango"
letter = predict_item[-1]
labled_letter = lb.transform([letter])
print(labled_letter)
print(clf.predict(labled_letter))

#Step 7
# Confusion matrix



