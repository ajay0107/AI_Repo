''' Machine learning in Python '''
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import numba
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
import random
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# load datasets from sklearn
from sklearn import datasets

iris=load_iris()

# keys of iris object
iris.keys()
n_samples,n_features = iris.data.shape
iris.data[0]

np.bincount(iris.target)
iris_data= iris.data

# plotting petal width
# histogram
x_index=3
for label in range(len(iris.target_names)):
    plt.hist(iris.data[iris.target==label,x_index],
         label=iris.target_names[label], alpha = 0.5)
plt.xlabel(iris.feature_names[3])
plt.legend(loc="upper right")

# Sepal length vs petal width
x_index=2
y_index = 3
for label in range(len(iris.target_names)):
    iris.data[iris.target==label,y_index]
    plt.scatter(x=iris.data[iris.target==label,x_index],
            y=iris.data[iris.target==label,y_index],
            label = iris.target_names[label])
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.legend(loc="upper left")

# create a dataframe
iris_rows=np.where(iris.target==0,iris.target_names[0],
                   np.where(iris.target==1,iris.target_names[1],
                   iris.target_names[2]))
iris_df = pd.DataFrame(iris.data, columns= iris.feature_names,
                       index=iris_rows)
pd.plotting.scatter_matrix(iris_df,c=iris.target,figsize=(8,8))

# load digits dataset
digits = datasets.load_digits()
digits.keys()
n_samples, n_features = digits.data.shape

#set up the figure
fig = plt.figure(figsize = (6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax=fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
    plt.imshow(digits.images[i],cmap=plt.cm.binary)
    # adding target labels
    ax.text(0,7,digits.target[i])

''' Exercise-1
laoding and plotting faces data '''

faces=datasets.fetch_olivetti_faces()

fig= plt.figure(figsize=(8,8))
for i in range(16):
    ax=fig.add_subplot(4,4,i+1,xticks=[],yticks=[])
    plt.imshow(faces.images[i],cmap=plt.cm.bone)

x,y=iris.data, iris.target
# splitting data into train and test data
train_x, test_x,train_y, test_y = train_test_split(x,y,train_size=0.5,
                                                   random_state=123,
                                                   stratify=y)
# training KNN classifier
classifier = KNeighborsClassifier().fit(train_x, train_y)
pred_y = classifier.predict(test_x)
print("Fraction accuracy")
print(np.sum(pred_y==test_y)/float(len(test_y)))

pred_y.shape
test_y.shape
correct_idx = np.where(pred_y==test_y)
incorrect_idx = np.where(pred_y!=test_y)
# plot two dimensions
for n in np.unique(test_y):
    # n = 0
    idx = np.where(test_y == n)
    plt.scatter(test_x[idx,1],test_x[idx,2],
                label=iris.target_names[n])
plt.scatter(test_x[incorrect_idx,1],test_x[incorrect_idx,2],
            color="darkred",label="incorrect ones")
plt.xlabel("Sepal width")
plt.ylabel("petal length")
plt.legend(loc=3)
plt.title("iris classification results")

























































































