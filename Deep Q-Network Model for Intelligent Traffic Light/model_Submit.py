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


class model_to_train:
    def __init__(self, total_layers, width, batch_size, alpha, layer_states, layer_actions):
        self._input_dim = layer_states
        self._output_dim = layer_actions
        self._batch_size = batch_size
        self._alpha = alpha
        self._model = self._build_model(total_layers, width)


    def _build_model(self, total_layers, width):
        """
        Build and compile a fully connected deep neural network

        """
        i=1
        inputs = keras.Input(shape=(self._input_dim,),name='input_layer')
        x = layers.Dense(width, activation='relu',name='hidden_layer'+str(i))(inputs)
        for _ in range(total_layers):
            i=i+1
            x = layers.Dense(width, activation='relu',name='hidden_layer'+str(i))(x)
        outputs = layers.Dense(self._output_dim, activation='linear',name='output_layer')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._alpha))
        return model
    

    def predict_neuron(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    def predict_batch_of_neurons(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1 ,verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def layer_states(self):
        return self._input_dim


    @property
    def layer_actions(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, layer_states, model_path):
        self._input_dim = layer_states
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_neuron(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    
    @property
    def layer_states(self):
        return self._input_dim