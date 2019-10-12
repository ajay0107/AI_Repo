# pandas tutorial- Scipy2019

###########################
# Chapter - 1- Intro
###########################

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
