# knn is used for classification - nearest neighbour algorithm
# always pick k to be an odd number to be able to classify data point into clusters

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# read data
data = pd.read_csv("KNN\car.data")
print(data.head())

# convert non numerical data to numerical in order to be able to analyse
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

# zip puts all our features into 1 list (buyimg,maint,door etc)- creates a tuple obj
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# split data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1)
# print(x_train, y_test)

# creating the model (5 neighbours in this case) play with neighbours and choose best
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("accuracy " + str(acc) + "\n")

# predict class
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: " + str(names[predicted[x]]) + ", Data: " +
          str(x_test[x]) + ", Actual: " + str(names[y_test[x]]) + "\n")
#    to see neighbours of points
    # n = model.kneighbors([x_test[x]], 9, True)
    # print("N: ", str(n))
