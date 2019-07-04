import math
import matplotlib
import numpy as np
import pandas as pd
# import seaborn as sns
import time
#
# from datetime import date, datetime, time, timedelta
from matplotlib import pyplot as plt
# from pylab import rcParams
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# # from tqdm import tqdm_notebook
#
# # %matplotlib inline
#
# #### Input params ##################
# stk_path = "googl.us.txt"
# test_size = 0.2                 # proportion of dataset to be used as test set
# cv_size = 0.2                   # proportion of dataset to be used as cross-validation set
# Nmax = 21                       # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
#                                 # Nmax is the maximum N we are going to test
# fontsize = 14
# ticklabelsize = 14
# ####################################


def predict_ma(df):

    l = len(df)
    if l < 4293:
        train = df[:round(l * 4 / 5)]
        test = df[round(l * 4 / 5):]
    else:
        train = df[-4293:l - 973]
        test = df[-973:]

    # Aggregating the dataset at daily level
    df['Timestamp'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df.Timestamp
    # df = df.resample('Close').mean()
    train['Timestamp'] = pd.to_datetime(train.Date, format='%Y-%m-%d')
    train.index = train.Timestamp
    # train = train.resample('Close').mean()
    test['Timestamp'] = pd.to_datetime(test.Date, format='%Y-%m-%d')
    test.index = test.Timestamp
    # test = test.resample('Close').mean()


    # # Plotting data
    # train.Close.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14)
    # test.Close.plot(figsize=(15, 8), title='Daily Ridership', fontsize=14)
    # plt.show()

    y_hat_avg = test.copy()
    y_hat_avg['moving_avg_forecast'] = train['Close'].rolling(60).mean().iloc[-1]
    plt.figure(figsize=(16, 8))
    plt.plot(train['Close'], label='Train')
    plt.plot(test['Close'], label='Test')
    plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
    plt.legend(loc='best')
    plt.show()

    return y_hat_avg['moving_avg_forecast']