
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time



class Nonlinear(tf.keras.Model):
    def __init__(self):
        super(Nonlinear, self).__init__()
        self.a = tf.Variable(-1., name='weight')
        self.b = tf.Variable(-1., name='bias')
        self.c = tf.Variable(-1., name='bias')
        self.d = tf.Variable(-1., name='bias')

    def call(self, inputs):
        return self.a*tf.pow(inputs,5)+self.b*tf.pow(inputs,3)+self.c*inputs+self.d


# toy dataset of points around 3*x+2
NUM_EXAMPLES = 30
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])*.05
training_outputs = tf.sin(training_inputs) + noise

# the loss function to be optimized
def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.a, model.b, model.c, model.d])



#plt.plot(training_inputs, training_outputs, 'o', color='black');

model = Nonlinear()
optimizer = tf.keras.optimizers.SGD(learning_rate=.02)

print(f"Initial loss: {loss(model, training_inputs, training_outputs)}")

xln = np.linspace(-2,2,500)

print('\tn\t|\ta\t\t|\tb\t\t|\tc\t\t|\td\t\t|\tError')
steps = 50
# a2dec = "{:.2f}".format(model.a.numpy())
# b2dec = "{:.2f}".format(model.b.numpy())
# loss2dec = "{:.2f}".format(loss(model, training_inputs, training_outputs))
# print(f"\t{0}\t|\t{a2dec}\t|\t{b2dec}\t|\t{loss2dec}")

yln = model(xln)
plt.plot(xln, yln, '-r', label=f"n={0}")
for i in range(steps):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.a, model.b, model.c, model.d]))
    if i % 1 == 0:
        a2dec = "{:.2f}".format(model.a.numpy())
        b2dec = "{:.2f}".format(model.b.numpy())
        c2dec = "{:.2f}".format(model.c.numpy())
        d2dec = "{:.2f}".format(model.d.numpy())
        loss2dec = "{:.2f}".format(loss(model, training_inputs, training_outputs))
        print(f"\t{i+1}\t|\t{a2dec}\t|\t{b2dec}\t|\t{c2dec}\t|\t{d2dec}\t|\t{loss2dec}")

        yln = model(xln)
        plt.plot(xln,yln,'-r', label=f"n={i}")

print(f"Final loss: {loss(model, training_inputs, training_outputs)}")
#print(f"A = {model.W.numpy()}, B = {model.B.numpy()}")

#plt.legend(loc='upper left')
plt.plot(training_inputs, training_outputs, 'o', color='black')

plt.show()

