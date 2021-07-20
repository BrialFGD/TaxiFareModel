from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data

import numpy as np
import pandas as pd

class Trainer:
   
    def __init__(self):
        pass

    # implement set_pipeline() function
    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        self.pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return self.pipe
    
    # implement train() function
    def run(self, X_train, y_train):
        self.set_pipeline()
        '''returns a trained pipelined model'''
        self.pipe.fit(X_train, y_train)
        return self.pipe
    
    # implement evaluate() function
    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = get_data(nrows=1000)
    df = clean_data(df)
    
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    
    New = Trainer()
    New.set_pipeline()
    New.run(X_train, y_train)
    rmse = New.evaluate(X_test, y_test)
    if rmse <= 8.989136548959618:
        print(f'model beats baseline with rmse={rmse}')
    else:
        print(f'model worst than baseline with rmse={rmse}')