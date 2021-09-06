
import tensorflow as tf


class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear, self).__init__()
        self.W = tf.Variable(5., name='weight')
        self.B = tf.Variable(10., name='bias')
    def call(self, inputs):
        return inputs * self.W + self.B


# toy dataset of points around 3*x+2
NUM_EXAMPLES = 2000
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



model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=.1)

print(f"Initial loss: {loss(model, training_inputs, training_outputs)}")

steps = 20
for i in range(steps):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]))
    if i % 20 == 0:
        print(f"Loss at step {i}: {loss(model, training_inputs, training_outputs)}")

print(f"Final loss: {loss(model, training_inputs, training_outputs)}")
print(f"W = {model.W.numpy()}, B = {model.B.numpy()}")



