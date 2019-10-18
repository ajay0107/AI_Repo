import os
import numpy as np
import pandas as pd
import pylab
from PIL import Image
from IPython.display import SVG
import matplotlib.pyplot as plt

params = {"legend.fontsize":"x-large",
          "figure.figsize":(15,5),
          "axes.labelsize":"x-large",
          "axes.titlesize":"x-large",
          "xtick.labelsize":"x-large",
          "ytick.labelsize" : "x-large"}

plt.rcParams.update(params)

import math
import timeit
from six.moves import cPickle as pickle
import platform
#from subprocess import check_output
import glob
import tensorflow as tf
import keras
from keras.constraints import maxnorm
#from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.utils.np_utils import to_categorical   
from keras.utils.vis_utils import model_to_dot
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from tqdm import tqdm_notebook
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# =============================================================================
# Set the GPU for use and check if it successful activate.
# =============================================================================
use_gpu = torch.cuda.is_available()
use_gpu

# opening and loading the file
def unpickle(fname):
    with open (fname, "rb") as f:
        result = pickle.load(f, encoding = "bytes")
    return result

def getData():
    labels_training = []
    dataImgSet_training = []
    labels_test = []
    dataImg_test = []
    # use "data_batch_*" for just the training set
    for fname in glob.glob("Deep_learning/data/cifar-10-batches-py/*data_batch*"):
    #  fname = 'Deep_learning/data/cifar-10-batches-py\\data_batch_1'
        print("getting data from:",fname )
        data = unpickle(fname)
        for i in range(10000):
            # i = 1
            img_flat = data[b'data'][i]
            labels_training.append(data[b'labels'][i])
            img_R = img_flat[0:1024].reshape(32,32)
            img_G = img_flat[1024:2048].reshape(32,32)
            img_B = img_flat[2048:3072].reshape(32,32)
            imgFormat = np.array([img_R, img_G,img_B])
            imgFormat = np.transpose(imgFormat,(1,2,0))
            dataImgSet_training.append(imgFormat)
        
        
    # use "test_batch_*" for just the test set
    for fname in glob.glob("Deep_learning/data/cifar-10-batches-py/*test_batch*"):
        #  fname = 'Deep_learning/data/cifar-10-batches-py\\test_batch'
        print("Getting data from : " , fname)
        data = unpickle(fname)
        for i in range(10000):
            # i = 0
            img_flat = data[b"data"][i]
            labels_test.append(data[b"labels"][i])
            img_R = img_flat[0:1024].reshape(32,32)
            img_G = img_flat[1024:2048].reshape(32,32)
            img_B = img_flat[2048:3072].reshape(32,32)
            imgFormat = np.array([img_R, img_G, img_B])
            imgFormat = np.transpose(imgFormat, (1, 2, 0))
            dataImg_test.append(imgFormat)
        
    dataImgSet_training = np.array(dataImgSet_training)
    labels_training = np.array(labels_training)
    dataImg_test =  np.array(dataImg_test)   
    labels_test = np.array(labels_test)

    return dataImgSet_training,labels_training,dataImg_test,labels_test
    
# data loading
X_train, y_train, X_test, y_test = getData()

labelNamesBytes = unpickle("Deep_learning/data/cifar-10-batches-py/batches.meta")
labelNames = []
for name in labelNamesBytes[b'label_names']:
    labelNames.append(name.decode("ascii"))
labelNames = np.array(labelNames)

fig = plt.figure(figsize = (6,6))
for i in range(0,9):
    # i = 1
    ax=fig.add_subplot(330+1+i)
    plt.imshow(Image.fromarray(X_test[i]))
    ax.set_title(labelNames[y_test[i]])
    
# =============================================================================
#  Using the raw values as input features might make models 
#  biased toward features having high magnitude values. 
#  It is still recommended to normalize and scale down the 
#  features, in our case, the images has a maximum of 255 
#  pixels, then it is sufficient divide the dataset by 
#  255.0 to normalize the data.    
# =============================================================================
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train =  X_train/255
X_test = X_test/255


# =============================================================================
# Remember there are 10 classes to work with and hence we are dealing with a multi-class classification problem, 
# then we need performs one hot encoding of the labels.
# =============================================================================

y_train =  to_categorical(y_train)
y_test = to_categorical(y_test)



    















