import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[.5, .5])
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                 [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5.,  10.])

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7
)

# incomplete tensor
mean = model.mean()

# evaluate inside of a session
with tf.compat.v1.Session() as sess:
    print(mean.numpy())

