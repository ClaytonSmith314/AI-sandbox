import tensorflow as tf


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


features_train = pd.read_csv('<SOME FILE LOCATION>')
features_eval = pd.read_csv('<SOME FILE LOCATION>')
label_train = features_train.pop('<LABEL_DATA>')
label_eval = features_eval.pop('<LABEL_DATA>')

COLLUMN_NAMES = ['one', 'two', 'three']
SPECIES = ['c1', 'c2']

# tensorflow objects require a tf.data.DataSet object to code data
def input_fn(data_df, label_df, num_epochs=10, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

feature_columns = []
for key in features_train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))


train_input_fn = input_fn(features_train, label_train)
eval_input_fn = input_fn(label_train, label_eval)

DNN_est = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30, 10],
    n_classes=3
)

DNN_est.train(
    input_fn=lambda: input_fn(features_train, label_train, training=True),
    steps=5000
)
# ^^^ PYTHON LAMBDAS!!!



# auto creates a learning model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result['accuracy'])

prediction = linear_est.predict(eval_input_fn)
print(prediction)
