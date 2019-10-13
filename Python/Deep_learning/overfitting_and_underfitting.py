from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results

# creating one-hot vector
train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)
plt.plot(train_data[2])

# creating the base-line model
base_line_model=keras.Sequential([
        keras.layers.Dense(16,activation="relu",input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16,activation = "relu"),
        keras.layers.Dense(1, activation = "sigmoid")
        ])
base_line_model.compile(optimizer = "adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy","binary_crossentropy"])

base_line_model.summary()
baseline_history=base_line_model.fit(train_data, train_labels,epochs=20,batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose = 2)
baseline_history.history.keys()
plt.plot(baseline_history.epoch,baseline_history.history["loss"],"r",
         label= " training loss of baseline model")
plt.plot(baseline_history.epoch,baseline_history.history["val_loss"],"g",
         label= " test loss of baseline model")
plt.legend()

# building model with less number of neurons
model_1=keras.Sequential([
        keras.layers.Dense(4,activation="relu",input_shape=(NUM_WORDS,)),
        keras.layers.Dense(4,activation = "relu"),
        keras.layers.Dense(1, activation = "sigmoid")
        ])
model_1.compile(optimizer = "adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy","binary_crossentropy"])

model_1.summary()
model_1_history=model_1.fit(train_data, train_labels,epochs=20,batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose = 2)
model_1_history.history.keys()
plt.plot(model_1_history.epoch,model_1_history.history["loss"],"r",
         label= " training loss of model_1 model")
plt.plot(model_1_history.epoch,model_1_history.history["val_loss"],"g",
         label= " test loss of model_1 model")
plt.legend()   
    
# building model with heavy number of neurons
model_2=keras.Sequential([
        keras.layers.Dense(512,activation="relu",input_shape=(NUM_WORDS,)),
        keras.layers.Dense(512,activation = "relu"),
        keras.layers.Dense(1, activation = "sigmoid")
        ])
model_2.compile(optimizer = "adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy","binary_crossentropy"])

model_2.summary()
model_2_history=model_2.fit(train_data, train_labels,epochs=20,batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose = 2)
model_2_history.history.keys()
plt.plot(model_2_history.epoch,model_2_history.history["loss"],"r",
         label= " training loss of model_2 model")
plt.plot(model_2_history.epoch,model_2_history.history["val_loss"],"g",
         label= " test loss of model_2 model")
plt.legend()

# plotting all in one graph
x= baseline_history.epoch
y=[baseline_history.history['binary_crossentropy'],baseline_history.history['val_binary_crossentropy'],
   model_1_history.history['binary_crossentropy'],model_1_history.history['val_binary_crossentropy'],
   model_2_history.history['binary_crossentropy'],model_2_history.history['val_binary_crossentropy']
   ]
labels = ["baseline train crossentropy","baseline test crossentropy",
          "small train crossentropy","small test crossentropy",
          "big train crossentropy","big test crossentropy"
          ]

color = ["r","--r","b","--b","g","--g"]
for y_arg,lab,col in zip(y,labels,color):
    plt.plot(x,y_arg,col,label=lab)
plt.legend()

# weight regularization
# L2 regularizer puts heavy penalty on large weights
baseregularizer_model=keras.Sequential([
        keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),
                           activation="relu",input_shape=(NUM_WORDS,)),
        keras.layers.Dense(16,kernel_regularizer=keras.regularizers.l2(0.001),
                           activation = "relu"),
        keras.layers.Dense(1, activation = "sigmoid")
        ])
baseregularizer_model.compile(optimizer = "adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy","binary_crossentropy"])

baseregularizer_model.summary()
baselineregularizer_history=baseregularizer_model.fit(train_data, train_labels,epochs=20,batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose = 2)
baselineregularizer_history.history.keys()

x= baseline_history.epoch
y=[baseline_history.history['binary_crossentropy'],baseline_history.history['val_binary_crossentropy'],
   baselineregularizer_history.history['binary_crossentropy'],baselineregularizer_history.history['val_binary_crossentropy'],
   ]
labels = ["baseline train crossentropy","baseline test crossentropy",
          "baselineregularizer train crossentropy","baselineregularizer test crossentropy",
          ]

color = ["r","--r","b","--b"]
for y_arg,lab,col in zip(y,labels,color):
    plt.plot(x,y_arg,col,label=lab)
plt.legend()

# Add dropout
# =============================================================================
# Dropout is one of the most effective and most commonly used regularization 
# techniques for neural networks, developed by Hinton and his students at the 
# University of Toronto. Dropout, applied to a layer, consists of randomly 
# "dropping out" (i.e. set to zero) a number of output features of the layer 
# during training. Let's say a given layer would normally have returned a 
# vector [0.2, 0.5, 1.3, 0.8, 1.1] for a given input sample during training; 
# after applying dropout, this vector will have a few zero entries distributed 
# at random, e.g. [0, 0.5, 1.3, 0, 1.1]. The "dropout rate" is the fraction of 
# the features that are being zeroed-out; it is usually set between 0.2 and 0.5.
#  At test time, no units are dropped out, and instead the layer's output values 
#  are scaled down by a factor equal to the dropout rate, so as to balance for 
#  the fact that more units are active than at training time.
# =============================================================================
addDropout_model=keras.Sequential([
        keras.layers.Dense(16,activation="relu",input_shape=(NUM_WORDS,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(16,activation = "relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation = "sigmoid")
        ])
addDropout_model.compile(optimizer = "adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy","binary_crossentropy"])

addDropout_model.summary()
addDropout_history=addDropout_model.fit(train_data, train_labels,epochs=20,batch_size=512,
                    validation_data=(test_data, test_labels),
                    verbose = 2)
addDropout_history.history.keys()

x= baseline_history.epoch
y=[baseline_history.history['binary_crossentropy'],baseline_history.history['val_binary_crossentropy'],
   addDropout_history.history['binary_crossentropy'],addDropout_history.history['val_binary_crossentropy'],
   ]
labels = ["baseline train crossentropy","baseline test crossentropy",
          "addDropout train crossentropy","addDropout test crossentropy",
          ]

color = ["r","--r","g","--g"]
for y_arg,lab,col in zip(y,labels,color):
    plt.plot(x,y_arg,col,label=lab)
plt.legend()




 
    
    
    
    
    
    
    
    
    
    
