from sklearn.ensemble import RandomForestRegressor
from lib import loadData, validateData, normalizeData, evaluateModel, visualizeData, visualizeLoss
from sklearn.metrics import mean_squared_error

(X, y, data_frame) = loadData()
validateData(data_frame)
(X_normalized, y_normalized, mean_X, mean_y, std_X, std_y) = normalizeData(X, y)


def forEachFold(X_train, y_train, X_test, y_test):
  model = RandomForestRegressor()
  model.fit(X_train, y_train.ravel())
  y_pred = model.predict(X_test)
  # Calculate Mean Squared Error (MSE) loss
  loss = mean_squared_error(y_test, y_pred)
  return loss

loss, mean_loss, std_loss = evaluateModel(X_normalized, y_normalized, forEachFold)
print(f"mean loss: {mean_loss} | std_loss: {std_loss}")

model = RandomForestRegressor()
model.fit(X_normalized, y_normalized.ravel())
visualizeData(model, data_frame, X, y, mean_X, mean_y, std_X, std_y)
# visualizeLoss(loss)
