import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm,preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from main import train_valid_split
import csv

def get_x_and_y(price,window_length=7,predict_day_length=1):
    '''get train and test set
    every time get window from price and
    '''
    m = len(price.iloc[0])
    n = len(price) - window_length
    m = window_length * m

    x = np.ones((n,m))
    y = np.ones((n,1))

    for i in range(len(price)-window_length):
        ans = [list(price.iloc[j] for j in range(i,i+window_length))]
        ans = np.array(ans).flatten()
        x[i] = ans
        y[i] = 1 if price.Close[i+window_length+predict_day_length-1] - price.Close[i+window_length-1] >0 else 0
    return [x,y]


def svm_prediction(stock_prices, x_train, y_train, x_valid, y_valid):


    # stock_prices = pd.read_csv(r'googl.us.txt')
    # stock_prices = pd.read_csv(r'NSE-TATAGLOBAL11.csv')
    # symbols = list(set(stock_prices['symbol']))
    #
    # msft_prices = stock_prices[stock_prices['symbol']== 'MSFT']

    #================================================
    stck_prices = stock_prices[['Date','Open','Low','High','Close']]
    stck_prices.to_csv('msft_prices.csv',sep='\t')
    dates = [pd.Timestamp(date) for date in stck_prices['Date']]

    close = np.array(stck_prices['Close'],dtype='float')

    plt.title('Google')
    plt.scatter(dates,close)
    plt.show()

    stck_prices = stck_prices.set_index('Date')



    # window_lengths = [7,14,21,30,60,90,120,150,180]
    accurarys = {}
    reports ={}

    # for l in window_lengths:
    #     print('window_length:',l)
    #     x, y = get_x_and_y(stck_prices, window_length=l)
    #     y = y.flatten()
    #     scaler = preprocessing.StandardScaler()
    #     scaler.fit_transform(x)
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=233)
    #     for kernel_arg in ['rbf', 'poly', 'linear']:
    #         clf = svm.SVC(kernel=kernel_arg, max_iter=5000)
    #         clf.fit(x_train, y_train)
    #         y_predict = clf.predict(x_test)
    #
    #         accurary = clf.score(x_test, y_test)
    #         report = classification_report(y_test, y_predict, target_names=['drop', 'up'])
    #         if l in accurarys:
    #             accurarys[l].append(accurary)
    #             reports[l].append(report)
    #         else:
    #             accurarys[l] = [accurary]
    #             reports[l] = [report]
    #         print('The Accurary of %s : %f' % (kernel_arg, clf.score(x_test, y_test)))
    #         print(report)


    # x, y = get_x_and_y(stck_prices, window_length=90)
    # y = y.flatten()
    # scaler = preprocessing.StandardScaler()
    # scaler.fit_transform(x)

    #============================================
    x_train, y_train, x_valid, y_valid, train, valid = train_valid_split(stck_prices)


    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=233)
    # for kernel_arg in ['rbf', 'poly', 'linear']:
    clf = svm.SVC(kernel='rbf', max_iter=5000)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_valid)

    # accurary = clf.score(x_test, y_test)
    report = classification_report(y_valid, y_predict, target_names=['drop', 'up'])
    # if l in accurarys:
    #     accurarys[l].append(accurary)
    #     reports[l].append(report)
    # else:
    #     accurarys[l] = [accurary]
    #     reports[l] = [report]
    # print('The Accurary of %s : %f' % ('rbf', clf.score(x_test, y_test)))
    print(report)

    return y_predict

if __name__ == '__main__':
    stock_prices = pd.read_csv('googl.us.txt')
    svm(stock_prices)