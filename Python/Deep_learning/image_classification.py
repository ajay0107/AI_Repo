from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
# import dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels)= fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
len(train_labels)

# preprocessing the data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)

# Normalizing the data
train_images = train_images/255
test_images = test_images/255

# plotting initial 25 images 
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# Build the model
# Building the neural network requires configuring 
# the layers of the model, then compiling the model.

# Defining NN model architecture
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128,activation=tf.nn.relu),
                          keras.layers.Dense(10, activation = tf.nn.softmax)
                          ])   

# Compiling the Neural Network
model.compile(optimizer =   tf.optimizers.Adam(),
              loss= "sparse_categorical_crossentropy",
              metrics=["accuracy"])

# training the network
model.fit(train_images,train_labels,epochs=5)

# now, we estimate prediction capability of our model on test data
pred_loss, pred_acc= model.evaluate(test_images, test_labels)    

# using trained model for prediction
prediction = model.predict(test_images)    
np.argmax(prediction[0])
test_labels[0]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    predicted_label=np.argmax(prediction[i])
    true_label=test_labels[i]
    if predicted_label==true_label:
        color="green"
    else: 
        color="red"
    plt.xlabel("{},({})".format(class_names[predicted_label],
               class_names[true_label]),
    color=color)

# adding more layers to increase accuracy of prediction
model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128,activation=tf.nn.relu),
                          keras.layers.Dense(128,activation=tf.nn.relu),
                          keras.layers.Dense(10, activation = tf.nn.softmax)
                          ])   



    
    











