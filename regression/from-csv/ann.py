import tensorflow as tf
from lib import loadData, validateData, normalizeData, evaluateModel, visualizeData, visualizeLoss

(X, y, data_frame) = loadData()
validateData(data_frame)
(X_normalized, y_normalized, mean_X, mean_y, std_X, std_y) = normalizeData(X, y)

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Lambda(lambda x: tf.concat([x, x**3], axis=1)),
  tf.keras.layers.Dense(units=5, activation='sigmoid'),
  tf.keras.layers.Dense(units=1)
])

def forEachFold(X_train, y_train, X_test, y_test):
  # reseting model weights
  tf.keras.backend.clear_session()
  # Compilar o modelo com uma taxa de aprendizado menor
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
  # Treinar o modelo
  model.fit(X_train, y_train, epochs=500, verbose=0)
  # Avaliar o modelo no conjunto de teste e armazenar a perda
  loss = model.evaluate(X_test, y_test)
  return loss

loss, mean_loss, std_loss = evaluateModel(X_normalized, y_normalized, forEachFold)
print(f"mean loss: {mean_loss} | std_loss: {std_loss}")

tf.keras.backend.clear_session()
# Compilar o modelo com uma taxa de aprendizado menor
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
# Treinar o modelo
model.fit(X_normalized, y_normalized, epochs=500, verbose=0)
visualizeData(model, data_frame, X, y, mean_X, mean_y, std_X, std_y)
# visualizeLoss(loss)


