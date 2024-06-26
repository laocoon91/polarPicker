import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Flatten, UpSampling1D
from tensorflow.keras.models import Model

# set variables
drop_rate = 0.3
learn_rate = 0.001

# -----------------------------------------------------------------------------
# get info on cpus, gpus
# -----------------------------------------------------------------------------

print(f'Using TensorFlow version {tf.__version__}')
# ? print(f'Using Keras version {tf.keras.__version__}')

cpus = tf.config.list_logical_devices("CPU")
gpus = tf.config.list_logical_devices("GPU")
print(f'Running on {len(cpus)} CPU(s)')
print(f'Running on {len(gpus)} GPU(s)')
if len(gpus) > 0:
    strategy = tf.distribute.MirroredStrategy(gpus)  # this works for 1 to multiple GPUs; but must create vars differently
    print('Using mirrored strategy to run on multiple GPUs on a single node')
else:
    strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
    print('Running on CPU or single GPU')

# -----------------------------------------------------------------------------
# define model
# -----------------------------------------------------------------------------

def build_polarPicker(drop_rate=drop_rate):
    """
    Define an autoencoder system
    """
    
    # encoder (uses timeseries as input)
    encoder = Sequential()
    encoder.add(Input(shape = (64,1)))
    encoder.add(Conv1D(32, 32, activation = 'relu', padding = 'same'))
    encoder.add(Dropout(drop_rate))
    encoder.add(BatchNormalization())
    encoder.add(MaxPooling1D(2, padding='same'))
    encoder.add(Conv1D(8, 16, activation = 'relu', padding = 'same'))
    encoder.add(BatchNormalization())
    encoder.add(MaxPooling1D(2, padding='same'))

    # decoder (uses encoder as input)
    decoder = Sequential()
    decoder.add(Input(shape = (16,8)))
    decoder.add(Conv1D(8, 16, activation = 'tanh', padding = 'same'))
    decoder.add(BatchNormalization())
    decoder.add(UpSampling1D(2))
    decoder.add(Conv1D(32, 32, activation = 'relu', padding = 'same'))
    decoder.add(BatchNormalization())
    decoder.add(UpSampling1D(2))
    decoder.add(Conv1D(1, 64, padding = 'same', activation = 'tanh'))

    # probability estimate (uses encoder as input)
    prob = Sequential()
    prob.add(Input(shape = (16,8)))
    prob.add(Flatten())
    prob.add(Dense(2, activation = 'softmax'))

    return encoder,decoder,prob

# -----------------------------------------------------------------------------
# Build model
# -----------------------------------------------------------------------------

encoder,decoder,prob = build_polarPicker()
encoder.summary()
decoder.summary()
prob.summary()

X = Input(shape = (64,1))
enc = encoder(X)
dec = decoder(enc)
p = prob(enc)

model = Model(inputs=X,outputs=[dec,p])

# loss is based on dec and p, using mean squared error and huber loss, respectively.
# dec loss is weighted 1, and p loss is weighted 200 (since we care more about p)
hub = tf.keras.losses.Huber(delta=0.5, name='huber_loss')

model.compile(optimizer = keras.optimizers.Adam(learning_rate=learn_rate), \
    loss=['mse', hub], loss_weights = [1,200],metrics = ['mse','acc'])

# -----------------------------------------------------------------------------
# Train model
# -----------------------------------------------------------------------------

# Training options:
# Stop the training early if no improvement after 15 epochs
# Reduce the learning rate by a factor of 10 if no improvement after 10 epochs.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,min_lr=1e-6)

# since model loss is determined by both the decoded signal and the estimated polarity,
# the output targets are the input xin and the polarity labels y.
history = model.fit(xin,[xin,y],validation_data=[vxin,vy],epochs=120,callbacks=[early_stop,reduce_lr])

# -----------------------------------------------------------------------------
# Save model
# -----------------------------------------------------------------------------

model.save("polarityModel.keras")

# -----------------------------------------------------------------------------
# Evaluate model
# -----------------------------------------------------------------------------

# validation data

# testing data
