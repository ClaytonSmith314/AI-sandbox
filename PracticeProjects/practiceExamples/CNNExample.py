
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# note, we can use augmentation to make the model better


MODEL_PATH = 'saves/CNNExampleModel.h5'

# load data set
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalize pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def image_show(index, is_test=False):  # doesn't work....
    if is_test:
        image = test_images[index],
        label = class_names[test_labels[index][0]]
    else:
        image = train_images[index],
        label = class_names[train_labels[index][0]]
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(label)
    plt.show()


# model = models.Sequential()

def load_model():
    model = models.load_model(MODEL_PATH)
    return model

# build model
def build_model():
    model = models.Sequential()

    # convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    print(model.summary())
    return model



def comp_train_save(model):
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=7,
                        validation_data=(test_images, test_labels))

    model.save(MODEL_PATH, save_format='h5')


# run once


