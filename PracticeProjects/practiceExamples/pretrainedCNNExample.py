
import tensorflow as tf

from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

import tensorflow_datasets as tfds

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,)

get_label_name = metadata.features['label'].int2str

def showImages(n, set):
    for image, label in set.take(n):
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label))

    plt.show()

# resizes the images so they are all the same size
IMG_SIZE = 160
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


# apply the resize function to each set
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,   # do we include the classifier part? No
                                                   weights='imagenet')

    base_model.trainable = False  # freezing the model. We don't want to train

    base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = keras.layers.Dense(1)

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    model.summary()

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def train_model(_model):
    history = _model.fit(train_batches,
                         epochs=1,
                         validation_data=validation_batches)

    acc = history.history['accuracy']
    print(acc)


MODEL_PATH = 'saves/catsAndDogs.h5'

create_model()

# model = models.load_model(MODEL_PATH)

#model.summary()



#train_model(model)
#model.evaluate(test_batches, verbose=1)

#model.save(MODEL_PATH, save_format='h5')


