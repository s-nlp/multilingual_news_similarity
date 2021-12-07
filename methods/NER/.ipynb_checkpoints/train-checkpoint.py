from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, GridSearchCV
import math
from tqdm import tqdm
import pandas as pd
import numpy as np


def dumb_classificator(x1, x2, x3):
    if np.isclose(x1, 0.0, atol=1e-02) and np.isclose(x2, 0.0, atol=1e-02) and np.isclose(x2, 0.0, atol=1e-02):
        return 1.0
    elif (x1 >= 0 and x1 < 0.06) or (x2 >= 0 and x2 < 0.06) or (x3 >= 0 and x3 < 0.06):
        return np.random.random() + 0.5
    elif (x1 >= 0.25) or (x2 >= 0.25) or (x3 >= 0.25):
        return np.random.random() + 3
    elif (x1 == -1.0) or (x2 == -1.0) or (x3 == -1.0):
        return np.random.random() + 3
    else:
        return 4.0


def train_eval(df):
    df.fillna(0.5, inplace = True)
    X = df[["sim_LOC", "sim_PER", "sim_ORG"]].values
    Y = df['Overall'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42, shuffle = True)
    
    models = [LinearRegression(), Lasso(), Ridge(), ElasticNet(), DecisionTreeRegressor(), KNeighborsRegressor(),
          GradientBoostingRegressor()]
    names = ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'DecisionTreeRegressor', 'KNeighborsRegressor',
              'GradientBoostingRegressor']
    
    res_d = {}
    for name, model in zip(names, models):
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)
        results = pd.DataFrame(np.array([pred, Y_test]).T, columns=["prediction", "similarity"])
        corr = results["prediction"].corr(results['similarity'])
        if not math.isnan(corr):
            res_d[name] = corr
            print(f"{name} correlation: {res_d[name]}")
        
    return res_d