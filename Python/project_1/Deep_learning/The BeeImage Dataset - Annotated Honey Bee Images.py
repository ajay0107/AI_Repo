#####################################
# Libraries
#####################################
# Common libs
import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path

# Image processing
import imageio
import skimage
import skimage.io
import skimage.transform
#from skimage.transform import rescale, resize, downscale_local_mean

# Charts
import matplotlib.pyplot as plt
import seaborn as sns


# ML
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics

#from sklearn.preprocessing import OneHotEncoder
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import tensorflow


# Set random seed to make results reproducable
np.random.seed(42)
tensorflow.set_random_seed(42)

# global variables 
img_folder = "Deep_learning/data/honey-bee-annotated-images/bee_imgs/bee_imgs/"
img_width = 100
img_height = 100
img_channels = 3

# =============================================================================
# Read Bee data
# =============================================================================
 bees = pd.read_csv("Deep_learning/data/honey-bee-annotated-images/bee_data.csv",
                    index_col = False,
                    parse_dates = {"datetime":[1,2]},
                    dtype = {'subspecies':'category', 'health':'category','caste':'category'})
 
# =============================================================================
#  Read and resize img, adjust channels. 
#  Caution: This function is not independent, it uses global vars: img_folder, img_channels
#  @param file: file name without full path
# =============================================================================

def read_img(file):
    # file = "001_043.png"
    img = skimage.io.imread(img_folder+file)
    img = skimage.transform.resize(img,(img_width, img_height),mode = "reflect")
    return img[:,:,:img_channels]

# dropping nas
bees.dropna(inplace=True)
img_exists = bees['file'].apply(lambda f: os.path.exists(img_folder + f))
bees = bees[img_exists]
bees.head()

# =============================================================================
# Distribution of bees by categories
# =============================================================================
f,ax=plt.subplots(nrows=2,ncols=2,figsize = (7,7))
bees.subspecies.value_counts().plot(kind="bar",ax=ax[0,0])
ax[0,0].set_ylabel('Count')
ax[0,0].set_title("SubSpecies")

bees.location.value_counts().plot(kind="bar",ax=ax[0,1])
ax[0,1].set_ylabel("location")
ax[0,1].set_title("count")

bees.caste.value_counts().plot(kind ="bar",ax=ax[1,0])
ax[1,0].set_ylabel("caste")
ax[1,0].set_title("count")

bees.health.value_counts().plot(kind="bar", ax= ax[1,1])
ax[1,1].set_ylabel("health")
ax[1,1].set_title("count")

f.subplots_adjust(hspace = 0.7)
f.tight_layout()


# =============================================================================
# Subspecies of Bee
# =============================================================================
subspecies = bees.subspecies.cat.categories
subspecies.values

















