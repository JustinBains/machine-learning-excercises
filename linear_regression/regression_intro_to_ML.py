""" https://www.youtube.com/watch?v=V59bYfIomVk&index=7&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v """

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


"""
1: intro to regression
2: Features and Labels   
3: Training and Testing
4: forecasting and predicting
5: Pickling and Scaling
"""


def data_frame():

    quandl.ApiConfig.api_key = '6qkzc9m2Sff_2X2yGQSx'
    df = quandl.get('WIKI/GOOGL')
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]  # these are features being used to calculate

    forecast_col = 'Adj. Close'
    df.fillna(-99999, inplace=True)
    num_days = 0.1  # this should be ratio of the number of days
    forecast_out = int(math.ceil(num_days*len(df)))  #
    print(forecast_out)

    df['label'] = df[forecast_col].shift(-forecast_out)

    # print(df.head())
    # print(df.tail())

    # X is features, y is labels
    X = np.array(df.drop(['label'], 1))  # everything is a feature except 'label' so they are all dropped
    X = preprocessing.scale(X)  # scales X before feeding it into the classifier, basically normalizes it
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    df.dropna(inplace=True)
    y = np.array(df['label'])

    """ TESTING AND TRAINING """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # clf = classifier
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)  # fit = train
    """ saves classifier as pickle """  # classifier is what is trained
    with open('linear_regression.pickle', 'wb') as f:
        pickle.dump(clf, f)

    pickle_in = open('linear_regression.pickle', 'rb')
    clf = pickle.load(pickle_in)

    accuracy = clf.score(X_test, y_test)  # score = test
    forecast_set = clf.predict(X_lately)

    print(forecast_set, accuracy, forecast_out)

    df['Forecast'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day
    for i in forecast_set:  # makes y variables (in date format
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

    df['Adj. Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


if __name__ == '__main__':
    data_frame()
