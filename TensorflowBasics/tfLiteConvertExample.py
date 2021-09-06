
import tensorflow as tf

# create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')  # comile the model
model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5)  # train the model
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir");

tf.saved_model.save(model, 'saves/normalModel')

# convert model
converter = tf.lite.TFLiteConverter.from_saved_model('saves/normalModel')
tflite_model = converter.convert()

# save the model
with open('saves/tflitemodel.tflite', 'wb') as f:
    f.write(tflite_model)
