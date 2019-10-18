import numpy as np
import pandas as pd
import pylab
from PIL import Image
from IPython.display import SVG
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image as imgk

inputPath = "Deep_learning/data/bird.jpg"

# =============================================================================
# We will focus on five main types of data augmentation 
# techniques for image data; specifically:
# =============================================================================

# =============================================================================
# Image shifts via the width_shift_range and height_shift_range arguments.
# Image flips via the horizontal_flip and vertical_flip arguments.
# Image rotations via the rotation_range argument
# Image brightness via the brightness_range argument.
# Image zoom via the zoom_range argument.
# =============================================================================

# =============================================================================
# For example, an instance of the ImageDataGenerator class can be constructed.
# =============================================================================

# create data generator
datagen = ImageDataGenerator()

# =============================================================================
# An iterator can be created from an image dataset loaded in memory 
# via the flow() function; for example:
# =============================================================================

# load the image
img = imgk.load_img(inputPath)
# convert image to numpy array 
data = imgk.img_to_array(img)

