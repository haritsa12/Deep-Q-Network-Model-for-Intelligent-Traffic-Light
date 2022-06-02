import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Define a simple sequential model

def create_model():

    inputs = keras.Input(shape=(80,))
    x = layers.Dense(512, activation='relu')(inputs)
    for _ in range(4):
        x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(4, activation='linear')(x) 

    model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
    model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=0.001))
    return model


# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()