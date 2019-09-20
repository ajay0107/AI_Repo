# matplotlib tutorial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x= np.linspace(0,10,100)
y1,y2,y3=np.cos(x),np.cos(x+1),np.cos(x+2)
names=["Signal 1","Signal 2","Signal 3"]

fig, ax = plt.subplots(nrows=3,ncols=1)
ax[0].plot(x,y1,color="black")
ax[1].plot(x,y2, color = "red")
ax[2].plot(x,y3,color="blue")

fig, ax = plt.subplots(nrows=3,ncols=1)
for a,y,name,col in zip(ax,[y1,y2,y3],names,["red","black","green"]):
    a.plot(x,y,color=col)
    a.set(xticks=[],yticks=[],title=name)
