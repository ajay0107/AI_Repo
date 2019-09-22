# pandas tutorial- Scipy2019

###########################
# Chapter - 1- Intro
###########################

import pandas as pd
import numpy as np
import seaborn as sns
import numba
import matplotlib.pyplot as plt
#import cv2
# from sklearn import linear_model
import sklearn as skl
# another way to import is
# from pandas import * # don't do this
pd.__format__
df=pd.read_csv("gapminder.tsv",sep="\t")

# to know the data type
type(df)
# Dimension of Datframe
df.shape
# Info of Dataframe
df.info()
# Head of dataframe
df.head()
# Tail of Dataframe
df.tail()
# Gives columns of dataframe
df.columns
# Gives indexes of dataframe
df.index
# Gives the value of Dataframe
df.values
# Gives only the datatype of columns
df.dtypes

# Gives mentioned column- in series format
# Gives only one column because it's a series
country=df["country"]
type(country)

# Gives mentioned columns- in dataframe format
# Gives multiple columns
country_df=df[["country","continent"]]
type(country_df)

# Drop multiple columns from dataframe
df_dropped_columns=df.drop(["country","continent"], axis="columns")

# Gives particular row- in series format
df.loc[1]
type(df.loc[1])
# Gives particular row- in dataframe format
df.loc[[0,1,2,3]]
type(df.loc[[0,1,2,3]])

# loc string match the entries with index

# Another method to extract particular row
# it matches the index position , doesn't string match with indexes
df.iloc[-1] # gives the first row from the end in series format
type(df.iloc[-1])
# in dataframe format
df.iloc[[0,1,2,3,-1,-2]]
type(df.iloc[[0,1,2,3,-1,-2]])

# Subsetting rows and columns together
subset = df.loc[:,["country","continent"]]
subset.head()

subset_1= df.loc[[0,1,2],["year","pop"]]
subset_1.head()

# Subsetting rows and columns together by iloc
# iloc matches index position, doesn't string match with indexes
subset = df.iloc[[0,1,2],[2,3]]
subset.head()

# Subsetting rows based on condition
subset_US = df.loc[df["country"]=="United States"] # OR
subset_US = df.loc[df["country"]=="United States",:]
subset_US.head()

# Subsetting rows based on multiple conditions
subset_US = df.loc[(df["country"]=="United States") & (df["year"]==1957),:]
subset_US.head()

# Groupby and Aggregate
df_groupby= df.groupby(["country","year"])[["lifeExp"]].mean()
type(df_groupby)
df

# Groupby and Aggregate - using aggregate function
# This has hierarchical index
df_groupby = df.groupby(["continent","year"])[["lifeExp","pop"]].agg(np.mean)
type(df_groupby)
df_groupby.head()

# To simplify hierarchical index into normal index
df_groupby=df_groupby.reset_index()

# Exerise - 1
tips = sns.load_dataset("tips")
tips.head()
tips.info()
tips.dtypes
# Unique values in a column of dataframe
tips.sex.unique()
tips.smoker.unique()
tips.day.unique()
tips.time.unique()

# Filter rows by smoker == 'No' and total_bill >= 10
subset_tips_1 = tips.loc[(tips["smoker"]=="No") & (tips["total_bill"]>=10),:]
type(subset_tips_1)
subset_tips_1.head()
subset_tips_1.shape

# average total_bill for each value of smoker, day, and time
avgTotalBillDf = tips.groupby(["smoker","day","time"])[["total_bill"]].agg(np.mean).reset_index()
avgTotalBillDf = avgTotalBillDf.loc[np.isfinite(avgTotalBillDf["total_bill"]),:]

###########################
# Chapter - 2-tidy
###########################

pew = pd.read_csv("pew.csv")
pew.head()
pew_tidy = pew.melt(id_vars="religion", var_name="income",
                    value_name="count")
# loading billboard dataset
billboard =  pd.read_csv("billboard.csv")
billboard.shape
billboard.info()
billboard_tidy = billboard.melt(id_vars=["year","artist","track","time", "date.entered"],
                                var_name="week", value_name="rank")
billboard_tidy.head()

# treating . as pipe operator
billboard_tidy = billboard.melt(id_vars=["year","artist","track","time", "date.entered"],
                                var_name="week", value_name="rank").groupby("artist")["rank"].mean()

# loading ebola dataset
ebola = pd.read_csv("country_timeseries.csv")
ebola.head()
ebola.shape
ebola_long = ebola.melt(id_vars=["Date","Day"], var_name="cd_country", value_name="count")
ebola_long.shape
ebola_long.head()
ebola_split = ebola_long["cd_country"].str.split("_", expand=True)
ebola_long[["status","country"]] = ebola_split

# loading weather dataset
weather= pd.read_csv("weather.csv")
weather.head()
weather_long = weather.melt(id_vars=["id","year","month","element"],
                            var_name="day",value_name="temp")
weather_long.head()
weather_wide =  weather_long.pivot_table(index=["id","year","month","day"],
                                    columns="element", values="temp", dropna=False)
weather_wide.head()
weather_wide = weather_wide.reset_index()

# Exerise - 2
tbl1= pd.read_csv("table1.csv")
tbl2= pd.read_csv("table2.csv")
tbl3= pd.read_csv("table3.csv")

# in table 3, just give the population
tbl3["population"]=tbl3["rate"].str.split("/",expand=True)[1]  # OR
tbl3["population"]=tbl3["rate"].str.split("/").str.get(1)

# Change table2 as wide format
tbl2_wide = tbl2.pivot_table(index=["country","year"], columns = "type", values="count")
tbl2_wide= tbl2_wide.reset_index()

###########################
# Chapter - 3- apply
###########################

# definition of function
def my_sq(x):
    return x**2
my_sq(9)
assert my_sq(4)==16

def my_avg(x,y):
    return((x+y)/2)
my_avg(4,6)

# creating new dataframe
newDf = pd.DataFrame({"a":[1,2,3],"b":[4,5,6]})
newDf["a"]**2
newDf.apply(my_sq)

def my_exp(x,e):
    return(x**e)
# apply function works columnwise taking single value at once, when axis = 0
newDf.apply(my_exp, e=3)

# It is error when function take more than 1 arguments
# columns of dataframe goes as a series
newDf.apply(my_avg)
# but when all arguments passed to function it works fine
newDf.apply(my_avg,y=4)

# numpy mean takes columns as argument
def avg_3(col):
    return(np.mean(col))
newDf.apply(avg_3)

# creating another function
# This functions doesn't take vectors as inputs
def avg_2_mod(x,y):
    if(x==20):
        return np.NaN
    else:
        return((x+y)/2)
avg_2_mod(newDf["a"], newDf["b"])

# Numpy helps function takes values as vector
avg_2_mod_vec=np.vectorize(avg_2_mod)
avg_2_mod_vec(newDf["a"], newDf["b"])

# avg_2_mod_vec can also be created by
@np.vectorize
def avg_2_mod_vec2(x,y):
    if(x==20):
        return np.NaN
    else:
        return((x+y)/2)
avg_2_mod_vec2(newDf["a"],newDf["b"])

# avg_2_mod_vec can also be created by
@numba.vectorize
def avg_2_mod_vec3(x,y):
    if(x==20):
        return np.NaN
    else:
        return((x+y)/2)
avg_2_mod_vec3(newDf["a"].values,newDf["b"].values)

# Exerise - 3
tbl3= pd.read_csv("table3.csv")
tbl3.dtypes
#create function
def extract_population(rate,sep="/",position=1):
    pop=rate.split(sep)[position]
    return int(pop)
tbl3["rate"].apply(extract_population)
tbl3["population"]=tbl3["rate"].apply(extract_population)

###########################
# Chapter - 4 - plots
###########################
tips["tip"].plot(kind="hist")
smokerCounts=tips["smoker"].value_counts()
smokerCounts.plot(kind="bar")

# Plotting using seaborn
sns.countplot(x="smoker",data=tips)
# Seaborn distribution plot
sns.distplot(tips["total_bill"])
# Seaborn linear Regression line
sns.lmplot(x="total_bill",y="tip", data=tips)
sns.lmplot(x="total_bill",y="tip", data=tips, hue = "sex")
# Plot without regression line
sns.lmplot(x="total_bill",y="tip", data=tips, hue = "sex", fit_reg=False,
           col="smoker")
sns.lmplot(x="total_bill",y="tip", data=tips, hue = "sex", fit_reg=False,
           col="smoker", row="day")
# Making grids/facet
facet = sns.FacetGrid(tips,col="time",row="smoker", hue="sex")
facet.map(plt.scatter,"total_bill","tip")

# Using matplotlib
plot = plt.subplots(2,1)
# Accessing individual plots
fig,(ax)=plt.subplots(1,1)
fig, (ax1,ax2)=plt.subplots(1,2)
ax1.scatter(tips["tip"],tips["total_bill"])
ax2.scatter(tips.loc[:,"time"],tips["total_bill"])

fig, (ax1, ax2)=plt.subplots(1,2)
sns.distplot(tips["tip"], ax=ax1)
sns.regplot(x="total_bill",y="tip", data=tips, ax=ax2)

# Exerise - 4
titanic = sns.load_dataset("titanic")
titanic.head()
fig, (ax1, ax2)=plt.subplots(1,2)
sns.distplot(titanic[["fare"]], ax=ax1)
sns.boxplot(x="class",y="fare",data=titanic, ax=ax2)

###########################
# Chapter - 5 - model
###########################
tips = sns.load_dataset("tips")
lr=skl.linear_model.LinearRegression()
lr.fit(X=tips[["total_bill","size"]],y=tips[["tip"]],)
lr.coef_
lr.intercept_

# Converting categorical variables in One Hot Encoding
# converting One-Hot-Encoding by Pandas
tips_one_hot=pd.get_dummies(tips)
lr_one_hot=skl.linear_model.LinearRegression()
lr_one_hot.fit(X=tips_one_hot.loc[:,tips_one_hot.columns[~tips_one_hot.columns.isin(["tips"])]],
                                                         y=tips_one_hot[["tip"]])
lr_one_hot.coef_
lr_one_hot.intercept_


##################################################################################
# Numpy Tutorial-Scipy2019
import numpy as np
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

# Matplotlib tutorial- Scipy2018

##########################################
# Chapter-1- Intro
##########################################
import matplotlib
print(matplotlib.__version__)
print(matplotlib.get_backend())

fig=plt.figure(facecolor=(1,0,0,0.1))
fig=plt.figure(figsize=plt.figaspect(0.1),facecolor=(1,0,0,0.1))

# creating a new axes
fig=plt.figure()
ax=fig.add_subplot(111)
ax.set(xlim=[0.5,4.5],ylim=[-2,8], title="Example Axes",
                   ylabel="Y-axis", xlabel="X-axis")
# creating new plot
# add_subplot
fig=plt.figure()
ax=fig.add_subplot(211)
ax.plot([1,2,3,4],[10,20,25,30],color="lightblue",linewidth=3)
ax.scatter([0.3,3.8,1.2,2.5],[11,25,9,26],c=[1,2,3,4],marker="^")
ax1=fig.add_subplot(2,1,2)
ax1.plot([1,2,3,4],[30,20,100,-1],color="black",linewidth=3)
ax1.scatter([0.3,3.8,1.2,2.5],[11,25,9,26],c=[1,2,3,4],marker="o")

# plt.subplots
fig, axes=plt.subplots(nrows=2, ncols=2)
axes[0,0].set(title="Upper Left")
axes[0,1].set(title="Upper Right")
axes[1,0].set(title="Lower Left")
axes[1,1].set(title="lower Right")
for ax in axes.flat:
    ax.set(xticks=[],yticks=[])

# Exercise - 1
x= np.linspace(0,10,100)
y1,y2,y3=np.cos(x),np.cos(x+1),np.cos(x+2)
fig,ax = plt.subplots(nrows=3, ncols=1)
ax[0].set(title="Signal-1")
ax[1].set(title="Signal-2")
ax[2].set(title="Signal-3")
ax[0].plot(x,y1,color="black",linewidth=3)
ax[1].plot(x,y2,color="black",linewidth=3)
ax[2].plot(x,y3,color="black",linewidth=3)
for i in ax.flat:
    i.set(xticks=[],yticks=[])

# another way of above
x= np.linspace(0,10,100)
y1,y2,y3=np.cos(x),np.cos(x+1),np.cos(x+2)
fig,ax = plt.subplots(nrows=3, ncols=1)
for axis,y,name in zip(ax,[y1,y2,y3],["Signal-1","Signal-2","Signal-3"]):
    axis.plot(x,y,color="black",linewidth=3)
    axis.set(title=name,xticks=[],yticks=[])

# bar plots
np.random.seed(1)
x=np.arange(5)
y=np.random.randn(5)
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=plt.figaspect(2))
ver_bars=axes[0].bar(x,y,color="green",align="center")
hori_bars=axes[1].barh(x,y,color="orange", align="center")
axes[0].axhline(0,color="gray",linewidth=2)
axes[1].axvline(0,color="gray",linewidth=2)

fig,ax=plt.subplots()
ver1_bars=ax.bar(x,y,color="green",align="center")
for bar, height in zip(ver1_bars,y):
    if height <0:
        bar.set(edgecolor="black", facecolor="salmon",linewidth=3)

fig,ax=plt.subplots()
ver1_bars=ax.bar(x,y,color="green",align="center")
for bar, height in zip(ver1_bars,y):
    if height <0:
        bar.set(edgecolor="black", color="salmon",linewidth=3)

fig,ax= plt.subplots()
y=np.random.randn(100).cumsum()
x=np.arange(100)
ax.fill_between(x,y,color="red")
ax.axhline(0,color="black", linewidth=2)

fig, axes=plt.subplots(nrows=2, ncols=1)
x=np.linspace(0,100,200)
y1=2*x+1
y2=3*x+1.2
y_mean=0.5*x*np.cos(2*x)+2.5*x+1.1
axes[0].fill_between(x,y1,y2, color="red")
axes[1].fill_between(x,y1,y_mean, color="blue")

fig, axes=plt.subplots(figsize=plt.figaspect(0.5))
x=np.linspace(0,100,200)
y1=2*x+1
y2=3*x+1.2
y_mean=0.5*x*np.cos(2*x)+2.5*x+1.1
axes.fill_between(x,y1,y2, color="yellow",zorder=1)
axes.plot(x,y_mean, color="blue",zorder=2)

# data object
x=np.linspace(0,100,200)
y1=2*x+1
y2=3*x+1.2
y_mean=0.5*x*np.cos(2*x)+2.5*x+1.1
fig, axes=plt.subplots(figsize=plt.figaspect(0.5))
dataObj={
        "x":x,
        "y1":2*x+1,
        "y2":3*x+1.2,
        "mean":0.5*x*np.cos(2*x)+2.5*x+1.1
        }
axes.fill_between("x","y1","y2", color="yellow", data=dataObj)
axes.plot("x","mean", color="black", data=dataObj)

# Exercise
''' y_raw=np.random.randn(1000).cumsum()+15
x_raw=np.linspace(0,24,y_raw.size)
x_pos=x_raw.reshape(10,100).min(axis=1)
y_avg = y_raw.reshape(10,100).mean(axis=1)
y_err=y_raw.reshape(10,100).ptp(axis=1)

bar_width = x_pos[1]-x_pos[0]
x_pred = np.linspace(0,30) '''
































































































