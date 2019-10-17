from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
imdb= keras.datasets.imdb
(train_data, train_labels),(test_data, test_labels)=imdb.load_data(num_words=10000)
print("training entries : {}, labels : {}".format(len(train_data),len(train_labels)))
print(train_data[0])

# dictionary mapping words to index
word_index=imdb.get_word_index()
# the first indices are reserved
word_index={k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"]=0
word_index["<START>"]=1
word_index["<UNK>"]=2 # unknown
word_index["<UNUSED>"]=3
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])
def decode_review(text):
    return " ".join([reverse_word_index.get(i,"2") for i in text])
decode_review(train_data[0])

# all reviews must be of same length for ANNs so we do padding
train_data=keras.preprocessing.sequence.pad_sequences(train_data,
                                                      value=word_index["<PAD>"],
                                                      padding = "post",
                                                      maxlen=256)
test_data=keras.preprocessing.sequence.pad_sequences(test_data,
                                                      value=word_index["<PAD>"],
                                                      padding = "post",
                                                      maxlen=256)
# Defining NN model architecture
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))
model.summary()

# Compiling the NN model
model.compile(optimizer = tf.optimizers.Adam(),
              loss= "binary_crossentropy",
              metrics=["accuracy"])

# splitting the data in train and validation set 
x_val=train_data[:10000]
partial_x_data =train_data[10000:]
y_val= train_labels[:10000]
partial_y_data = train_labels[10000:]

# training the network
history = model.fit(partial_x_data, partial_y_data,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val,y_val),
                    verbose = 1 )
# Evaluating the model
results = model.evaluate(test_data, test_labels)

# History of model fitting
history_dict= history.history
history_dict.keys()
train_accuracy=history_dict["accuracy"]
validation_accuracy=history_dict["val_accuracy"]
train_loss=history_dict["loss"]
validation_loss=history_dict["val_loss"]
epochs = range(1,len(train_accuracy)+1)
# plotting accuracy
plt.plot(epochs,train_accuracy, "bo", label="Training accuracy")
plt.plot(epochs, validation_accuracy,"b", label ="Validation accuracy")
plt.legend()
# plotting loss
plt.plot(epochs,train_loss, "bo", label="Training loss")
plt.plot(epochs, validation_loss,"b", label ="Validation loss")
plt.legend()

# prediction of test data
prediction = model.predict(test_data)
prediction[prediction < 0.5] = 0
prediction[prediction >= 0.5]= 1



















