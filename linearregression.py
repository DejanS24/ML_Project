import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os



def load_frames():
    df1 = pd.DataFrame()
    i = 0

    for filename in os.listdir('stocks'):
        if os.stat("stocks/" + filename).st_size != 0:
            i = i+1
            if i < 100:
                df = pd.read_csv('stocks/' + filename)
                df1 = df1.append(df)

    return df1,i


def configure_dataset(df1,i):
    split_len = int(0.7* i)
    print(df1)
    df1.sort_values(by = ['Date'], inplace=True, ascending=True)
    train = df1[:split_len]
    test = df1[split_len:]
    X_train = train.drop(['Date', 'OpenInt',  'Close'],axis=1)
    y_train = train[['Close']]
    helper = test.drop(['Close', 'OpenInt'], axis = 1)
    X_test  = test.drop(['Date', 'OpenInt', 'Close'],axis=1)
    y_test = test[['Close']]
    return X_train, y_train, helper, X_test, y_test

'''
X = df1.drop(['Date', 'OpenInt', 'Volume', 'Close'])
y = df1[['Close']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

'''


def run_regression(X_train, y_train, X_test, y_test):

   # X_train, y_train, helper, X_test, y_test = configure_dataset(df1, df1.__len__())

    regressor = LinearRegression()

    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)


   # helper['Close'] = y_pred

    showResults(y_test,y_pred)

    return y_pred


def showResults(y_test, y_pred):
  #  print(helper)
    print("R2 score: {}".format(r2_score(y_test, y_pred)))
    print("MSE score: {}".format(mean_squared_error(y_test, y_pred)))
    rms=np.sqrt(np.mean(np.power((np.array(y_test)-np.array(y_pred)),2)))
    print(rms)

if __name__ == "__main__":
    run_regression()
