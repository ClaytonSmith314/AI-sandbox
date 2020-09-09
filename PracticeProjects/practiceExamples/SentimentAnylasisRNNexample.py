
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 64),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    print(model.summary())

    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['acc'])

    return model


MODEL_PATH = 'saves/MovieReviews.h5'

model = create_model()
# model.load(MODEL_PATH)

history = model.fit(train_data, train_labels, epochs=25, validation_split=0.2, verbose=1)

model.save(MODEL_PATH)
