# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 19:54:21 2020
Utility functions for financial analysis to prepare X and Y matrices
    1. get_model_data((dbInput, dbList, ric, horiz, drop_rows, triggerUp1 = 0, triggerLoss = 0, triggerUp2 = 0)) => return dataX, dataY
    2. get_model_Y(dataset, horizon, triggerUp1 = 0, triggerLoss = 0, triggerUp2 = 0) => return dataY
    3. get_droprows(dataset, num_rows) => return dataset    Remove rows in dataset improper due to tech indicators (usually 50 to 200 rows)
    4. get_cleandupli_dataset(dataset) => return dataset
    5. get_model_cleanXset(X_raw, trigger) => return X_clean   Remove columns where number of missing values exceeds a trigger
    6. get_train_test_resize(dataX, dataY, proportionTrain) => return (X_train, y_train), (X_test, y_test)
    7. get_features_ident_xgb(dataX, proportionTrain) => return (X_train, y_train), (X_test, y_test)
    
    
@author: JDE65
"""
import pandas as pd
import numpy as np
import sqlite3 as sq
import matplotlib.pyplot as plt
from itertools import chain
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf


###--- 1. Load Data for any dense NN
def get_model_data(dbInput, dbList, ric, horiz, drop_rows, triggerUp1 = 0, triggerLoss = 0, triggerUp2 = 0):
    connect = sq.connect(dbInput)        # connect to the DB with the stock price information ###
    cursor = connect.cursor()
    dataRIC = 'data' + ric
    tableforRIC = "SELECT * FROM "
    dbRIC = tableforRIC + dataRIC
    cursor.execute(dbRIC)
    outp = cursor.fetchall()
    dataTrad = pd.DataFrame(outp, columns =['index', 'reindex', 'instrument', 'date', 'close', 'open', 'high', 'low', 'dvt', 'volume', 'dtr', '1wktr', '1mthtr', '3mthtr'])

    dataTrad.drop(dataTrad[pd.isnull(dataTrad["close"])].index, inplace=True)           # clean DB by dropping all rows where closing price is null
    dataTrad.drop(dataTrad[pd.isnull(dataTrad["open"])].index, inplace=True)            # clean DB by dropping all rows where closing price is null
    dataTrad = dataTrad.drop('index', 1)                                                           # clean DB by dropping all rows where closing price is null
    dataTrad = dataTrad.drop('reindex', 1)                                                         # clean DB by dropping all rows where closing price is null
    dataTrad['date'] = dataTrad['date'].str[:4] + dataTrad['date'].str[5:7] + dataTrad['date'].str[8:10]       #convert long string in 8 character string for date formating
    dataTrad['date'] = dataTrad['date'].astype(int)
    dataTrad['1wktr'] = dataTrad['1wktr'].astype(float)
    dataTrad['1mthtr'] = dataTrad['1mthtr'].astype(float)
    dataTrad['3mthtr'] = dataTrad['3mthtr'].astype(float)
    dataX = dataTrad
    dataY = get_model_Y(dataTrad, horiz, triggerUp1, triggerLoss, triggerUp2 = 0)
    
    return dataX, dataY


###--- 2. Compute Y matrix from a dataset based on horizon and gain-loss triggers ---###
def get_model_Y(dataset, horizon, triggerUp1 = 0, triggerLoss = 0, triggerUp2 = 0):
    dataY = pd.DataFrame(columns = ['date', 'yval', 'returnF','futClose'])
    dataY['date'] = dataset['date']
    dataY['yval'] = dataY['yval'].astype(float)                                # initialize y value
    dataY['returnF'] = dataY['returnF'].astype(float)
    dataY['futClose'] = dataY['futClose'].astype(float)
    #dataY['yval'] = 0                                # initialize y value
    #dataY['returnF'] = 0                             #reinitialize returnF (future return value)
    for i in range(len(dataset)):
        if (i <= (len(dataset) - horizon - 1)):                                                      # if row above the last number of rows for which indicators are not meaningfull
            futClose = dataset.at[i + horizon, 'close']    
            perfFut = (futClose / dataset.at[i + 1, 'open']) - 1
        else:
            perfFut = 0

        dataY.at[i, 'returnF'] = perfFut
        dataY.at[i, 'futClose'] = futClose
        if (perfFut < triggerLoss):
            dataY.at[i, 'yval'] = -1
        elif (perfFut >= triggerUp1):
            if (triggerUp2 >0 and perfFut >= triggerUp2):
                dataY.at[i, 'yval'] = 2
            dataY.at[i, 'yval'] = 1
        else:
            dataY.at[i, 'yval'] = 0
    # print(i , perfFut, dataY.at[i,'returnF'], dataY.at[i,'yval'])   === Print the Y matrix for debugging
    return dataY
    #print('Matrix Y computed')


###--- 3. Drop unnecessary rows into a dataset + summary ---###
def get_droprows(dataset, num_rows):
    dataset = dataset.drop(dataset.index[0:num_rows])
    return dataset

###--- 4. Clean up a dataset for duplicates & fill in NaN ---###
def get_cleandupli_dataset(dataset):
    dataset = dataset.drop_duplicates()         #remove duplicated entries
    #dataset = dataset.fillna(method='ffill')    # fill NaN forward with next valid observation - opposite method = bfill 
    
    return dataset

###--- 5. Cleanup X : remove row with missing values above a trigger ---###
def get_model_cleanXset(X_raw, trigger):
    X_raw = X_raw.iloc[:,3:]
    nanc = X_raw.isna().sum()
    for i in range(len(nanc)):
        ananc = nanc.iloc[i]/len(X_raw)
        if ananc > trigger:
            X_raw = X_raw.drop(columns = i)
    X_clean = X_raw.T.reset_index(drop = True).T
    return X_clean



###--- 6. Split X and Y dataset into X_train X_test y_train y_test ---###
def get_train_test_price(dataX, dataY, nn_start, nn_size, proportionTrain):
    Xs = np.array(dataX)
    ys = np.array(dataY)
    X = Xs[:,:]           # dataX with 'instrument', 'date', 'close', 'open', ...
    res = ys[:,2] * 1               # array with effective return over the horizon
    futClose = ys[:,3] * 1          # array with future close of the stock
    y = futClose * 1
    
    y[np.isnan(y)] = 0      # convert nan into 0 for y
    X[np.isnan(X)] = 0      # convert nan into 0 for X
    
    if nn_start > 0:
        X = X[nn_start:, :]
        y = y[nn_start:,]
        res = res[nn_start:]
        
    if X.shape[0] > nn_size:           # apply the split on the lowest of lstmSize or X length
        train_size = int(nn_size * proportionTrain)
    else:
        train_size = int(X.shape[0] * proportionTrain)
    X_train = X[:train_size, :]
    X_test = X[train_size: nn_size, :]

    y_train = y[:train_size]
    y_test = y[train_size: nn_size]
    res_train = res[:train_size]
    res_test = res[train_size: nn_size]
    return (X_train, y_train), (X_test, y_test), (res_train, res_test)
    

###--- 6. Split X and Y dataset into X_train X_test y_train y_test ---###
def get_train_test_return(dataX, dataY, nn_start, nn_size, proportionTrain, modeRC):
    Xs = np.array(dataX)
    ys = np.array(dataY)
    X = Xs[:,:]           # dataX with 'instrument', 'date', 'close', 'open', ...
    res = ys[:,2] * 1               # array with effective return over the horizon
    futClose = ys[:,3] * 1          # array with future close of the stock
    y = res * 1
    if modeRC == 'class':           # if classification => if return >=0 then buy, else let
        y[y >= 0] = 1   
        y[y < 0.1] = 0
        
    y[np.isnan(y)] = 0      # convert nan into 0 for y
    X[np.isnan(X)] = 0      # convert nan into 0 for X
    
    if nn_start > 0:
        X = X[nn_start:, :]
        y = y[nn_start:,]
        res = res[nn_start:]
        
    if X.shape[0] > nn_size:           # apply the split on the lowest of lstmSize or X length
        train_size = int(nn_size * proportionTrain)
    else:
        train_size = int(X.shape[0] * proportionTrain)
    X_train = X[:train_size, :]
    X_test = X[train_size: nn_size, :]

    y_train = y[:train_size]
    y_test = y[train_size: nn_size]
    res_train = res[:train_size]
    res_test = res[train_size: nn_size]
    
    return (X_train, y_train), (X_test, y_test), (res_train, res_test)
    




###--- 2. Scale X_train X_test on X_train ---###
def get_model_scaleX(X_train, X_test):
    train_mean = np.mean(X_train, axis = 0)
    train_std = np.std(X_train, axis = 0)
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    return (X_train, X_test), (train_mean, train_std)


###--- 9. Split X matrix into X & Y with train & test for xgb_validation ---###

def get_features_ident_xgb(dataX, proportionTrain):
    y = dataX['close']      # y is the closing price
    X = dataX.iloc[:,3:]    # X are all other elements of the X matrix
    
    train_size = int(X.shape[0] * proportionTrain)
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    
    return (X_train, y_train), (X_test, y_test)   
    



###--- 11. To be developped ---###

def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)
    
    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['mav05'],label='MA 5', color='g',linestyle='--')
    plt.plot(dataset['price'],label='Closing Price', color='b')
    plt.plot(dataset['mav20'],label='MA 20', color='r',linestyle='--')
    plt.plot(dataset['bollUpperband'],label='Upper Band', color='c')
    plt.plot(dataset['bollLowerband'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['bollLowerband'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for the dataset - last {} days.'.format(last_days))
    plt.ylabel('EUR')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['macd'],label='MACD', linestyle='-.')
    plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    plt.plot(dataset['log_momentum'],label='Momentum', color='b',linestyle='-')

    plt.legend()
    plt.show()

   