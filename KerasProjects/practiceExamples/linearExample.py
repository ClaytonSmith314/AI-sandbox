import tensorflow as tf


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


features_train = pd.read_csv("<SOME FILE LOCATION>")
features_eval = pd.read_csv("<SOME FILE LOCATION>")
label_train = features_train.pop('<LABEL_DATA>')
label_eval = features_eval.pop('<LABEL_DATA>')

CATEGORICAL_COLLUMNS = ['one', 'two', 'three']
NUMERIC_COLLUMNS = ['c1', 'c2']

feature_columns = []
for feature_name in CATEGORICAL_COLLUMNS:
    vocabulary = features_train[feature_name].unique()
    feature_columns.append(tf.feature_column.catigorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# tensorflow objects require a tf.data.DataSet object to code data
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function


train_input_fn = make_input_fn(features_train, label_train)
eval_input_fn = make_input_fn(label_train, label_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)

# auto creates a learning model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print(result['accuracy'])

prediction = linear_est.predict(eval_input_fn)
print(prediction)




