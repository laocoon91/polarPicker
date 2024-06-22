import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
import tensorflow as tf
from tf import keras
from keras import layers
from keras.models import Model
from layers import Input, LSTM
#from keras import backend as K

# Define model
drop_rate = 0.3
learn_rate = 0.001

X = Input(shape = (64,1))
x = layers.Conv1D(32, 32, activation = 'relu', padding = 'same')(X)
x = layers.Dropout(drop_rate)(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(2, padding='same')(x)
x = layers.Conv1D(8, 16, activation = 'relu', padding = 'same')(x)
x = layers.BatchNormalization()(x)
enc = layers.MaxPooling1D(2, padding='same')(x)

x = layers.Conv1D(8, 16, activation = 'tanh', padding = 'same')(enc)
x = layers.BatchNormalization()(x)
x = layers.UpSampling1D(2)(x)
x = layers.Conv1D(32, 32, activation = 'relu', padding = 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling1D(2)(x)
dec = layers.Conv1D(1, 64, padding = 'same', activation = 'tanh')(x)

x_flat = layers.Flatten()(enc)

p = layers.Dense(2, activation = 'softmax')(x_flat)

# model input is X, outputs are dec and p
model = Model(X, [dec, p])

# loss is based on dec and p, using mean squared error and huber loss, respectively.
# dec loss is weighted 1, and p loss is weighted 200 (since we care more about p)
hub = tf.keras.losses.Huber(delta=0.5, name='huber_loss') 
#model.compile(optimizer = 'adam', loss=['mse', hub], loss_weights = [1,200],metrics = ['mse','acc'])
model.compile(optimizer = keras.optimizers.Adam(learning_rate=learn_rate), \
    loss=['mse', hub], loss_weights = [1,200],metrics = ['mse','acc'])

# why is learning rate set on the backend? UPDATE: Looks deprecated, using different method when compiling
# K.set_value(model.optimizer.learning_rate, learn_rate)

# Train model

# Training options:
# Stop the training early if no improvement after 15 epochs
# Reduce the learning rate by a factor of 10 if no improvement after 10 epochs. 
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,min_lr=1e-6)

# since model loss is determined by both the decoded signal and the estimated polarity,
# the output targets are the input xin and the polarity labels y.
trained = model.fit(xin,[xin,y],validation_data=[vxin,vy],epochs=100,callbacks=[early_stop,reduce_lr])

# Performance metrics?

# save model
model.save("polarityModel.keras")
