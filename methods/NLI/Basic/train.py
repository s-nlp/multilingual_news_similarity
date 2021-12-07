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


def train_eval(df):
    X = df[["entailment", "neutral", "contradiction"]].values
    Y = df['Overall'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42, shuffle = True)
    
    models = [LinearRegression(), Lasso(), Ridge(), ElasticNet(), DecisionTreeRegressor(**{'criterion': 'mae', 'max_depth': 60}), KNeighborsRegressor(),
          GradientBoostingRegressor(**{'learning_rate': 0.1, 'max_depth': 11, 'n_estimators': 500})]
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