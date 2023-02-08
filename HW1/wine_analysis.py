import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

df_red = pd.read_csv('winequality-red.csv', sep=';') 
df_white = pd.read_csv('winequality-white.csv', sep=';')

target_column = ['quality'] 
predictors = list(set(list(df_red.columns))-set(target_column))
df_red[predictors] = df_red[predictors]/df_red[predictors].max()

# X = df_red[predictors].values
# y = df_red[target_column].values
X_w = df_white.drop(columns = 'quality')
y_w = df_white['quality']
X_r = df_red.drop(columns = 'quality')
y_r = df_red['quality']

X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_w, y_w)
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_r, y_r)
print(X_w_train.shape); print(X_w_test.shape)

#ridge for white
rr_w = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10.0, 100.0])
rr_w.fit(X_w_train, y_w_train)
print("score: ", rr_w.score(X_w_train, y_w_train))
pred_train_rr_w = rr_w.predict(X_w_train)
print('W Ridge RMSE: ', mean_squared_error(y_w_train,pred_train_rr_w, squared=False))
print('W Ridge MSE: ', mean_squared_error(y_w_train,pred_train_rr_w))
print('W Ridge r2: ',r2_score(y_w_train, pred_train_rr_w))

#ridge for red
rr_r = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10.0, 100.0])
rr_r.fit(X_r_train, y_r_train)
print('Score: ', rr_r.score(X_r_train, y_r_train))
pred_train_rr_r = rr_r.predict(X_r_train)
print('R Ridge RMSE: ', mean_squared_error(y_r_train,pred_train_rr_r, squared=False))
print('R Ridge MSE: ', mean_squared_error(y_r_train,pred_train_rr_r))
print('R Ridge r2: ',r2_score(y_r_train, pred_train_rr_r))

# pred_test_rr = rr.predict(X_test)
# print('Ridge RMSE: ', mean_squared_error(y_test,pred_test_rr, squared=False))
# print('Ridge MSE: ', mean_squared_error(y_test,pred_test_rr))
# print('Ridge r2: ', r2_score(y_test, pred_test_rr))

#lasso
model_lasso_w = Lasso(alpha=0.01)
model_lasso_w.fit(X_w_train, y_w_train) 
pred_train_lasso_w= model_lasso_w.predict(X_w_train)
print('W Lasso MRSE: ', mean_squared_error(y_w_train,pred_train_lasso_w, squared= False))
print('W Lasso RSE: ', mean_squared_error(y_w_train,pred_train_lasso_w))
print('W Lasso r2: ',r2_score(y_w_train, pred_train_lasso_w))

model_lasso_r = Lasso(alpha=0.01)
model_lasso_r.fit(X_r_train, y_r_train) 
pred_train_lasso_r= model_lasso_r.predict(X_r_train)
print('R Lasso MRSE: ', mean_squared_error(y_r_train,pred_train_lasso_r, squared= False))
print('R Lasso RSE: ', mean_squared_error(y_r_train,pred_train_lasso_r))
print('R Lasso r2: ',r2_score(y_r_train, pred_train_lasso_r))

# pred_test_lasso= model_lasso.predict(X_test)
# print('Lasso MRSE: ', mean_squared_error(y_train,pred_train_lasso, squared= False))
# print('Lasso RSE: ', mean_squared_error(y_train,pred_train_lasso))
# print('Lasso r2: ',r2_score(y_train, pred_train_lasso))

#linear
lr_w  = LinearRegression()
lr_w.fit(X_w_train, y_w_train)
pred_train_lr_w= lr_w.predict(X_w_train)
print('W Linear MRSE: ', mean_squared_error(y_w_train,pred_train_lr_w, squared= False))
print('W Linear RSE: ', mean_squared_error(y_w_train,pred_train_lr_w))
print('W Linear r2: ', r2_score(y_w_train, pred_train_lr_w))

lr_r  = LinearRegression()
lr_r.fit(X_r_train, y_r_train)
pred_train_lr_r= lr_r.predict(X_r_train)
print('R Linear MRSE: ', mean_squared_error(y_r_train,pred_train_lr_r, squared= False))
print('R Linear RSE: ', mean_squared_error(y_r_train,pred_train_lr_r))
print('R Linear r2: ', r2_score(y_r_train, pred_train_lr_r))

# pred_test_lr= lr.predict(X_test)
# print(np.sqrt(mean_squared_error(y_test,pred_test_lr))) 
# print(r2_score(y_test, pred_test_lr))

# dummy_reg_w = DummyRegressor()
# dummy_reg_w.fit(X_w_train, y_w_train)

# y_pred_w = dummy_reg_w.predict(X_w_train)
# mse_w = mean_squared_error(y_w_test, y_pred_w)
# rmse_w = np.sqrt(mse_w)
# print("W Dummy Constant RMSE:", rmse_w)

# dummy_reg_r = DummyRegressor()
# dummy_reg_r.fit(X_r_train, y_r_train)

# y_pred_r = dummy_reg_r.predict(X_r_train)
# mse_r = mean_squared_error(y_r_test, y_pred_r)
# rmse_r = np.sqrt(mse_r)
# print("W Dummy Constant RMSE:", rmse_r)