from sklearn.linear_model import LinearRegression
from lib import loadData, validateData, normalizeData, evaluateModel, visualizeData, visualizeLoss
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


(X, y, data_frame) = loadData()
validateData(data_frame)
(X_normalized, y_normalized, mean_X, mean_y, std_X, std_y) = normalizeData(X, y)

def forEachFold(X_train, y_train, X_test, y_test):
  model = LinearRegression()
  poly = PolynomialFeatures(degree=5, include_bias=False)
  poly_features = poly.fit_transform(X_train.reshape(-1, 1))

  model.fit(poly_features, y_train)
  y_pred = model.predict(poly.fit_transform(X_test.reshape(-1, 1)))
  # Calculate Mean Squared Error (MSE) loss
  loss = mean_squared_error(y_test, y_pred)
  return loss

def transformInput(input): 
  poly = PolynomialFeatures(degree=5, include_bias=False)
  poly_features = poly.fit_transform(input.reshape(-1, 1))
  return poly_features

loss, mean_loss, std_loss = evaluateModel(X_normalized, y_normalized, forEachFold)

print(f"mean loss: {mean_loss} | std_loss: {std_loss}")

model = LinearRegression()
poly = PolynomialFeatures(degree=5, include_bias=False)
poly_features = poly.fit_transform(X_normalized.reshape(-1, 1))
model.fit(poly_features, y_normalized)

visualizeData(model, data_frame, X, y, mean_X, mean_y, std_X, std_y, transformInput)
# visualizeLoss(loss)
