import tensorflow as tf
from regression import loadData, validateData, normalizeData, evaluateModel, visualizeData

(X, y, data_frame) = loadData()
validateData(data_frame)
(X_normalized, y_normalized, mean_X, mean_y, std_X, std_y) = normalizeData(X, y)

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Lambda(lambda x: tf.concat([x, x**3], axis=1)),
  tf.keras.layers.Dense(units=5, activation='sigmoid'),
  tf.keras.layers.Dense(units=1)
])

evaluateModel(model, X_normalized, y_normalized)
visualizeData(model, data_frame, X, y, mean_X, mean_y, std_X, std_y)

