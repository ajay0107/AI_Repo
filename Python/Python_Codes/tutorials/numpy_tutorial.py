
##################################################################################
# Numpy Tutorial-Scipy2019
import pandas as pd
import numpy as np
import seaborn as sns
import numba
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
import random
#import cv2
# from sklearn import linear_model
import sklearn as skl
#create np array- all data in array must be of same type
a=np.array([1,2,3,4])
type(a)
a.dtype

a[0]
a[1]

# assign new value
a[0]=10
#assign float number in integer array
a[0]=12.6 # doesn't round off

# Dimensionality of array
a.ndim
# shape gives rows and columns of array
a.shape
# size gives total number of elements in array
a.size
# numpy arrays gives elementwise operations of array
d=np.array([56,57,58,59])
a+d
d/a
d*a
d**a
a*10
a**5

# Universal functions(ufunc)
np.sin(a)
# Multidimensional array
a=np.array([[0,1,2,3],[5,6,7,8]])
b=np.array([[0,1,2,3],[5,6,7,8],["a","b","c","d"]])
# b is 2D array here
b.ndim
b.shape
b.size
# Accessing particular element of array
b[2,3]
b[2,3]="good"
# Access to all columns of particular row in array
b[2]

# slicing an array
a1=np.array([10,11,12,13,14,15])
a1[2:4]
a1[0:-2]
a1[-4:3]

# grab first three elements
a1[:3]
# grab last two elements
a1[-2:]
# goes from start to end with stride of 2
a1[::2] # var[lower:upper:step]

a2=np.array([range(0,6),range(11,17),range(50,56),range(100,106)])
a2[1:,4:]
a2[:,5]
a2[::2,::2]

a1=np.arange(25).reshape(5,5)
red=a1[:,[1,3]] # OR
red=a1[:,1::2]
yellow=a1[-1,:]
blue=a1[1::2,:3:2]

# slicing doesn't create new array, it only looks at particular part of array
b1=a1[1,:]
b1[4]= 1000
# also changed a1
a1

# create copy of any array
b1=a1.copy()
b1=b1[1,:]
b1[4]= 150
# doesn't change a1
a1
# inserting data in array
b1=a1.copy()
b1[2,-2:] = [112,142]
b1[2,-2:] = 845

# Exercise-1
img = plt.imread("Numpy_Tutorial_SciPyConf_2019_master/exercises/filter_image/dc_metro.png")
plt.imshow(img, cmap=None)
def smooth(img):
    avg_img=  ( img[:-2,1:-1] #Top
             + img[2:,1:-1]  # Bottom
             + img[1:-1,:-2] # Left
             + img[1:-1,2:]  # Right
             + img[1:-1,1:-1])/5 # centre
    return(avg_img)

avg_img=smooth(img)
for num in range(50):
    avg_img=smooth(avg_img)
plt.imshow(avg_img)

# Logical indexing or Fancy indexing
a=np.arange(0,80,10)
indices = [1,2,-3]
y=a[indices]
a[indices]=99
# logical condition to select elements of array
mask = np.array([0,1,1,0,0,1,0,0],dtype=bool)
a[mask]
a = np.array([0,-1,3,4,-5,6,2,-6,-3,-9])
mask=a<0
a[mask]
a[mask]=120
test=np.arange(25).reshape(5,5)
test[0,2]
test[[0,2,-2,-2],[2,3,1,-1]]
mask=test%3==0
test[mask]

# Computation with arrays
a=np.ones((3,5))
b1=np.ones((5))
b=np.ones((5))
b.reshape(1,5)
a.shape
b1.shape
b1.ndim
b.ndim
b.shape
c=a+b
a=np.ones((5,1))
b1=np.ones((5,))
c=a+b1

# Aggregate methods
a=np.array([[1,2,3],[5,6,7],[7,8,9],[12,24,14]])
a.sum()
a.sum(axis=0) # columnwise
a.sum(axis=1) # rowwise
a.sum(axis=-2) # -1 gives last axis , which is always rowwise
# sequence of axis is that first - rowwise, second-columnwise,
# then new dimensional gets first place

# min=max
a= np.array([[1,2,3],[4,5,6]])
a.max(axis=0)
a.max(axis=1)
a=np.array([[1,2,3],[100,6,7],[-9,8,9],[12,24,14]])
a.argmax(axis=0)
a.argmax(axis=1)
a.argmax() # returns the location in 1D
np.unravel_index(a.argmax(),a.shape) # returns location in multi-dimensions

# where clause
# 1D where clause
a=np.array([1,2,3,4,5,6,5,6])
a==a.max()
np.where(a==a.max())
# 2D where clause
t = np.array([[1,2,3,'bar'],
 [2,3,4,'bar'],
 [5,6,7,'hello'],
 [8,9,1,'bar']])
rows, cols = np.where(t == 'bar')

# Exercise-2
wind_data = np.loadtxt('Numpy_Tutorial_SciPyConf_2019_master/exercises/wind_statistics/wind.data')

# calculate the min, max and mean windspeeds and standard deviation of the
# windspeeds over all the locations and all the times

winds=wind_data[:,3:]
Dates=wind_data[:,:3]
max1=winds.max()
min1=winds.min()
std1=winds.std()

# calculate the min, max and mean windspeeds and standard deviations of the
#   windspeeds at each location over all the days (a different set of numbers
#   for each location)
max2=winds.max(axis=0)
min2=winds.min(axis=0)
std2=winds.std(axis=0)

#Calculate the min, max and mean windspeeds and standard deviations of the
#  windspeeds at each location over all the days (a different set of numbers
# for each location)
max3=winds.max(axis=1)
min3=winds.min(axis=1)
std3=winds.std(axis=1)

# Find the location which has the greatest windspeed on each day (an integer
#   column number for each day)
max4=winds.argmax(axis=1)
max4.shape
# Find the year, month and day on which the greatest windspeed was recorded
max5=np.where(winds==winds.max())
var=Dates[max5[0]]
# another way to doing above- line 514
max6 = np.unravel_index(winds.argmax(),winds.shape)
var=Dates[max6[0]]
# Find the average windspeed in January across all years, for each location
datejan=np.where(Dates[:,1]==1)
datejan=datejan[0]
avgJan=winds[datejan,:]
avgJan=avgJan.mean(axis=0)

# difference between flatten and ravel
a=np.array([[0,1,2,3],[19,6,7,8]])
b1=a.flatten()
b1[0]=100
# flatten doesn't change a
b2=a.ravel()
b2[0]=100









































































