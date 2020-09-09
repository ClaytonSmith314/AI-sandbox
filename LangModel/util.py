
from random import Random as rand
import random
from LangModel import objects as ob
import tensorflow as tf

BASE_SIZE = 128

# holds utility methods needed for the model


# memory
def load_all(directory):
    pass


def save_all(directory):
    pass




# creation
def rand_string(n=16, letters='abcdefghijklmnopqrstuvwxyz0123456789'):
    key = ''.join(random.choice(letters) for i in range(n))
    return key


def converging_dnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(shape=[128, 2]),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid),  # input layer
        tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid),  # hidden layer
        tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid)  # output layer (4)
    ])
    return model


def flat_dnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid),  # input layer
        tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid),  # hidden layer
        tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid)  # output layer (4)
    ])
    return model


def diverging_dnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid),  # input layer
        tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid),  # hidden layer
        tf.keras.layers.Dense(128, activation=tf.keras.activations.sigmoid)  # output layer (4)
    ])
    return model


def stems(n=64):
    methods = {}
    for i in range(n):
        methods[rand_string()] = converging_dnn_model()
    for i in range(n):
        methods[rand_string()] = flat_dnn_model()
    for i in range(n):
        methods[rand_string()] = diverging_dnn_model()
    return methods

        

