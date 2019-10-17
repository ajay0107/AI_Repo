from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import unidecode
import time
print(tf.__version__)

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text=unidecode.unidecode(open(path_to_file).read())
len(text)

# unique contains all unique characters in file 
unique = sorted(set(text))
# creating mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(unique)}
idx2char = {i:u for i, u in enumerate(unique)}

# setting maximum length for input as characters
max_length=100
# length of vocabulary in chars
vocab_size = len(unique)
# Number of RNNS (here GRU) units
units = 1024
# batch size 
batch_size = 64
# buffer size to shuffle our dataset
buffer_size = 10000












