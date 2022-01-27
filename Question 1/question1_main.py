import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from question1 import regression

dataset = pd.read_csv('diabetes.csv')

# first 10 rows of dataset
print(dataset.head(10))

# see how many of datas are in class 1 and how many in class 0
print("outcomes:")
print(dataset['Outcome'].value_counts())

# split dataset to train set and test set
data = pd.DataFrame(dataset,columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']);
X = data.drop("Outcome", axis=1)
Y = data.drop(data.columns[[0,1,2,3,4,5,6,7]], axis=1)
data = StandardScaler().fit_transform(data)
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


y_train_array = y_train.values
y_test_array = y_test.values

# class regression from logistic python file
reg = regression()

# number of features
features = x_train.shape[1]

print("Number of features:")
print(features)

# initializing the weight array with 0
w = np.zeros((1,features))
b = 0

# fit the best w and b for classification with decent gradient
wb, gradient, costs = reg.fit(w, b, x_train, y_train_array, epochs=4500,lr=0.0001)

# print optimize w and b after fit
w = wb["w"]
b = wb["b"]
print("Optimized w:")
print(w)
print('Optimized b:')
print(b)

# final prediction
z_train = reg.sigmoid(np.dot(w,x_train.T)+b)
z_test = reg.sigmoid(np.dot(w,x_test.T)+b)


train_r = x_train.shape[0]
test_r = x_test.shape[0]

#if z>0.5 predictied class is 1,else 0
y_tr_pred = reg.predict(z_train, train_r)
print('accuracy of train:')
train_acc = accuracy_score(y_tr_pred.T, y_train_array)
print(train_acc)
# accuracy of test set
y_ts_pred = reg.predict(z_test, test_r)
print("accuracy of test:")
test_acc = accuracy_score(y_ts_pred.T,y_test_array)
print(test_acc)

# error rates
print("error rate of train set: ")
print(1-train_acc)
print("error rate for test set:")
print(1-test_acc)

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Cost reduction over time')
plt.show()