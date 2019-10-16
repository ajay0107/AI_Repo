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
        img_flat = data[b'data'][i]
        labels_training.append(data[b'data'][2])
    
    
















