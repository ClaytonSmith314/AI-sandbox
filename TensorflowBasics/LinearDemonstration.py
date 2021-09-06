
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time


class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear, self).__init__()
        self.W = tf.Variable(-1., name='weight')
        self.B = tf.Variable(-1., name='bias')
    def call(self, inputs):
        return inputs * self.W + self.B


# toy dataset of points around 3*x+2
NUM_EXAMPLES = 15
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# the loss function to be optimized
def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])



plt.plot(training_inputs, training_outputs, 'o', color='black');

model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=.15)

print(f"Initial loss: {loss(model, training_inputs, training_outputs)}")

xln = np.linspace(-2,2,2)

print('\tn\t|\ta\t\t|\tb\t\t|\tError')
steps = 10
a2dec = "{:.2f}".format(model.W.numpy())
b2dec = "{:.2f}".format(model.B.numpy())
loss2dec = "{:.2f}".format(loss(model, training_inputs, training_outputs))
print(f"\t{0}\t|\t{a2dec}\t|\t{b2dec}\t|\t{loss2dec}")

yln = training_inputs.numpy() * model.W.numpy() + model.B.numpy()
plt.plot(training_inputs, yln, '-r', label=f"n={0}")
for i in range(steps):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 1 == 0:
        a2dec = "{:.2f}".format(model.W.numpy())
        b2dec = "{:.2f}".format(model.B.numpy())
        loss2dec = "{:.2f}".format(loss(model, training_inputs, training_outputs))
        print(f"\t{i+1}\t|\t{a2dec}\t|\t{b2dec}\t|\t{loss2dec}")

        yln = training_inputs.numpy() * model.W.numpy() + model.B.numpy()
        plt.plot(training_inputs,yln,'-r', label=f"n={i}")

print(f"Final loss: {loss(model, training_inputs, training_outputs)}")
print(f"A = {model.W.numpy()}, B = {model.B.numpy()}")

#plt.legend(loc='upper left')
plt.show()

