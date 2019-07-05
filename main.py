import pandas as pd
import numpy as np

import calendar

import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

from pyramid.arima import auto_arima

from sklearn.preprocessing import MinMaxScaler

# from fbprophet import Prophet
scaler = MinMaxScaler(feature_range=(0, 1))


def load_file(filename):
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']
    data = df.sort_index(ascending=True, axis=0)
    # print(df.head())
    #plt.figure(figsize=(16, 8))
    #plt.plot(df['Close'], label='Close Price history')
    return data


def preprocess(data):
    new_data = pd.DataFrame(index=range(0, len(data)),
                                columns=['Close', 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                                         'Month_start_end', 'Week_start_end'])
    for i in range(0, len(data)):
        new_data['Close'][i] = data['Close'][i]
        new_data['Year'][i] = data['Date'][i].year
        new_data['Month'][i] = data['Date'][i].month
        new_data['Week'][i] = data['Date'][i].isocalendar()[1]
        new_data['Day'][i] = data['Date'][i].day
        new_data['Dayofweek'][i] = data['Date'][i].weekday()
        new_data['Dayofyear'][i] = data['Date'][i].timetuple().tm_yday
        if (data['Date'][i].day - 1 == calendar.monthrange(data['Date'][i].year, data['Date'][i].month)[0]) or \
                (data['Date'][i].day - 1 == calendar.monthrange(data['Date'][i].year, data['Date'][i].month)[1]):
            new_data['Month_start_end'][i] = 1
        else:
            new_data['Month_start_end'][i] = 0
        if (data['Date'][i].weekday() == 0) or (data['Date'][i].weekday() == 6):
            new_data['Week_start_end'][i] = 1
        else:
            new_data['Week_start_end'][i] = 0
            
    return new_data


def train_valid_split(new_data):
    l = len(new_data)
    if l < 4293:
        train = new_data[:round(l * 4 / 5)]
        valid = new_data[round(l * 4 / 5):]
    else:
        train = new_data[-4293:l - 973]
        valid = new_data[-973:]

    x_train = train.drop('Close', axis=1)
    y_train = train['Close']
    x_valid = valid.drop('Close', axis=1)
    y_valid = valid['Close']

    # x_train_scaled = scaler.fit_transform(x_train)
    # x_train = pd.DataFrame(x_train_scaled)
    # x_valid_scaled = scaler.fit_transform(x_valid)
    # x_valid = pd.DataFrame(x_valid_scaled)

    return x_train, y_train, x_valid, y_valid, train, valid


def knn_predict(x_train, y_train, x_valid):
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
    knn = neighbors.KNeighborsRegressor()
    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train, y_train)
    preds = model.predict(x_valid)
    # print(x_train)
    # print(y_train)
    return preds


def calculate_rmse(y_valid, preds):
    return np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)), 2)))


def plot_graph(train, valid, preds):
    valid['Predictions'] = 0
    valid['Predictions'] = preds
    plt.plot(valid[['Close', 'Predictions']])
    plt.plot(train['Close'])
    plt.show()


def auto_arima_predict(data):

    l = len(data)
    if l < 4293:
        train = data[:round(l * 4 / 5)]
        valid = data[round(l * 4 / 5):]
    else:
        train = data[-4293:l - 973]
        valid = data[-973:]

    training = train['Close']
    validation = valid['Close']

    model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                       trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(training)

    forecast = model.predict(n_periods=round(len(data)*1/5))
    forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

    plt.plot(train['Close'])
    plt.plot(valid['Close'])
    plt.plot(forecast['Prediction'])
    plt.show()

    print(np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(forecast['Prediction'])), 2))))

    return forecast['Prediction']


def prophet_predict(train, valid):
    # model = Prophet()
    # model.fit(train)
    #
    # close_prices = model.make_future_dataframe(periods=len(valid))
    #
    # return model.predict(close_prices)
    return []

if __name__ == "__main__":

    data = load_file("ko.us.txt")

    new_data = preprocess(data)

    x_train, y_train, x_valid, y_valid, train, valid = train_valid_split(new_data)

    preds = knn_predict(x_train, y_train, x_valid)

    print(calculate_rmse(y_valid, preds))

    plot_graph(train, valid, preds)

    auto_arima_predict(data)





