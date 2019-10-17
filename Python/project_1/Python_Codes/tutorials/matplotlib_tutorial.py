# Matplotlib tutorial- Scipy2018

##########################################
# Chapter-1- Intro
##########################################
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


=======
# 2D arrays or images
data= np.random.randn(225).reshape(15,15)
fig,ax=plt.subplots()
im = ax.imshow(data, cmap="gist_earth")
fig.colorbar(im)

data= np.random.randn(225).reshape(15,15)
fig,ax=plt.subplots()
im = ax.imshow(data, cmap="seismic")
fig.colorbar(im)

t= np.arange(0,5,0.2)
plt.plot(t, t, t, t**2, t, t**3)
# color specification for plots
plt.plot(t,t,"red",t,t**2,"green",t,t**3,"blue")
# markers
t=np.arange(0,5,0.1)
plt.plot(t,t,"o",t,t**2,"+",t,t**3,":")
# combining markers with colors
plt.plot(t,t,"*y",t,t**2,"8m",t,t**3,"sb")
# lineStyles
t=np.arange(0,5,1)
plt.plot(t,t,":",t,t**2,"-",t,t**3,"-.")
# Plot attributes
t=np.arange(0,5,0.2)
plt.plot(t,t,"y:",t,t**2,"b-",t,t**3,"r-.")

t =np.arange(0,5,0.1)
a=np.exp(-t)*np.cos(2*np.pi*t)
plt.plot(t,a,"r:D",mfc="yellow",mec="green")



