import os
import tensorflow as tf
import cProfile
import matplotlib.pyplot as plt


# fetch and format the mnist data
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis]/255, tf.float32),
    tf.cast(mnist_labels, tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

# build the model
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu',
                           input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

# call the model and inspect the output in eager execution (not keras built in model)
for images, labels in dataset.take(1):
    print("Logits: ", mnist_model(images[0:1]).numpy())


# training loop with eager execution
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_history = []


def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)

        tf.debugging.assert_equal(logits.shape, (32, 10))

        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))


def train(epochs):
    for epoch in range(epochs):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(images, labels)
        print(f'Epoch {epoch} finished')


train(epochs=3)

plt.plot(loss_history, 'bo')
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')

plt.show()











