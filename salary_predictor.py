import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("emp_sal.csv")
x = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values
# SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel='poly', degree=4, gamma="auto")  # fixed 'kerne' to 'kernel'
svr_reg.fit(x, y)

svr_model_pred = svr_reg.predict([[6.5]])
print(svr_model_pred)
# Knn
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=5, weights="distance", algorithm='auto', p=2)
knn_reg.fit(x, y)

knn_model_pred = knn_reg.predict([[6.5]])
print(knn_model_pred)

# Decision tree 
from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(criterion="friedman_mse",
                             splitter='random',
                             max_depth=4,
                             min_samples_split=5,
                             random_state=0)
dt_reg.fit(x,y)

decision_tree_model_pred=dt_reg.predict([[6.5]])
print(decision_tree_model_pred)
# RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=50,
                           criterion="friedman_mse", 
                           max_depth=4,min_samples_split=5,
                           random_state=0)
rf_reg.fit(x,y)

random_forest_model_pred=rf_reg.predict([[6.5]])
print(random_forest_model_pred)
