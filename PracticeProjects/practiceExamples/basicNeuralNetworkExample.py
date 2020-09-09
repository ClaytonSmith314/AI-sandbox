
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

class_names = ['T_shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Angle boot']

# setup data set
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



# preprocessing
train_images = train_images / 255.0  # rescale the values to between 0 and 1
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),   # input layer (1)
    keras.layers.Dense(128, activation='relu'),   # hidden layer (2)
    keras.layers.Dense(10, activation='softmax')  # output layer (4)
])

print(type(train_images))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# model.fit(train_images, train_labels, epochs=3)
# model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

print('Test accuracy: ', test_acc)

show = 25

predictions = model.predict(test_images)
prediction = model.predict(test_images[1].reshape([1, 28, 28]))
print(prediction)

print('number\tValue\t\tprediction')
for n in range(show):
    print(str(n)+'\t\t'+class_names[test_labels[n]]+'\t\t'+class_names[np.argmax(predictions[n])])

def show_image(n):
    plt.figure()
    plt.imshow(test_images[n])
    plt.colorbar()
    plt.grid(False)
    plt.show()
