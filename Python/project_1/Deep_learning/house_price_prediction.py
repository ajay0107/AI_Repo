from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

boston_housing = keras.datasets.boston_housing
(train_data, train_labels),(test_data, test_labels)=boston_housing.load_data()

# shuffle the training data
order=np.argsort(np.random.random(train_labels.shape))
train_data=train_data[order]
train_labels=train_labels[order]

# normalizing the data
mean=train_data.mean(axis=0)
std=train_data.std(axis=0)
train_data = (train_data- mean)/std
test_data = (test_data-mean)/std

# model building function
def build_model():
    model= keras.Sequential([
            keras.layers.Dense(64,activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
            keras.layers.Dense(64, activation = tf.nn.relu),
            keras.layers.Dense(1)
            ])
    optimizer = tf.optimizers.RMSprop(0.001)
    model.compile(loss="mse",
                  optimizer = optimizer,
                  metrics = ["mae"])
    return model

model= build_model()
model.summary() 

# display training process by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 ==0 : print("")
        print(".",end="")
    
EPOCHS=500

# store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2,verbose =0, callbacks=[PrintDot()])
plt.plot(history.epoch,history.history["mae"],"r", label = "MAE of training data")
plt.plot(history.epoch,history.history["val_mae"],"b", label = "MAE of validation data")
plt.legend()
plt.plot(history.epoch,history.history["loss"],"r", label = "loss of training data")
plt.plot(history.epoch,history.history["val_loss"],"b", label = "loss of validation data")
plt.legend()

# defining early stopping callback to stop training if model doesn't improve
model= build_model()
early_stop = keras.callbacks.EarlyStopping(monitor="val_mae",patience=20)
history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2,
                    verbose =0, callbacks=[early_stop,PrintDot()])
 
plt.plot(history.epoch,history.history["mae"],"r", label = "MAE of training data")
plt.plot(history.epoch,history.history["val_mae"],"b", label = "MAE of validation data")
plt.legend()

# evaluate model on test dataset
loss, mae = model.evaluate(test_data, test_labels, verbose = 0)
print(loss, mae*1000)

# predictions of test_dataset
test_predictions = model.predict(test_data)





















    














