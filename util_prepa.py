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

###--- 1. Drop unnecessary rows into a dataset + summary ---###
def get_droprows(dataset, num_rows, horiz):
    dataset = dataset.drop(dataset.index[0:num_rows])
    dataset = dataset.drop(dataset.index[(len(dataset)-horiz):])
    return dataset

###--- 2. Clean up a dataset for duplicates & fill in NaN ---###
def get_cleandupli_dataset(dataset):
    dataset = dataset.drop_duplicates()         #remove duplicated entries
    #dataset = dataset.fillna(method='ffill')    # fill NaN forward with next valid observation - opposite method = bfill 
    return dataset

###--- 3. Cleanup X : remove row with missing values above a trigger ---###
def get_cleanXset(X_raw, trigger):    # ONLY if only basic data but no enriched data
    X_raw = X_raw.iloc[:,2:]   
    nanc = X_raw.isna().sum()
    for i in range(len(nanc)):
        ananc = nanc.iloc[i]/len(X_raw)
        if ananc > trigger:
            X_raw = X_raw.drop(columns = i)
    X_clean = X_raw.T.reset_index(drop = True).T
    return X_clean

def get_cleanXset_full(X_raw, trigger):
    X_raw = X_raw.iloc[:,4:]
    nanc = X_raw.isna().sum()
    for i in range(len(nanc)):
        ananc = nanc.iloc[i]/len(X_raw)
        if ananc > trigger:
            X_raw = X_raw.drop(columns = i)
    X_clean = X_raw.T.reset_index(drop = True).T
    return X_clean

###--- 4. Cleanup X : remove row with missing values above a trigger ---###
def get3D_create_dataset(dataX, dataY, seq_length, step):
    Xs, ys = [], []
    for i in range(0, len(dataX) - seq_length + 1, step):
        v = dataX.iloc[i:(i + seq_length)].values
        Xs.append(v)
        ys.append(dataY.iloc[i + seq_length - 1])
    return np.array(Xs), np.array(ys)

###--- 6. Split X and Y dataset into X_train X_test y_train y_test ---###
## 6.1 Base case - Price 
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
    
##--- 6.2. Create sliding window and enrich dataX and dataY ---###
def get3D_train_test_price(dataX, dataY, nn_start, nn_size, proportionTrain):
    X = dataX[:, :, :]    # dataX with 'instrument', 'date', 'close', 'open', ...
    res = dataY[:,2] * 1      # separating the effective return between train & test
    futClose = dataY[:,3] * 1
    y = futClose * 1     # y = close price => regression on predicting future price
    
    if nn_start > 0:
        X = X[nn_start:, :, :]
        y = y[nn_start:,]
        res = res[nn_start:]
        
    if X.shape[0] > nn_size:           # apply the split on the lowest of lstmSize or X length
        train_size = int(nn_size * proportionTrain)
    else:
        train_size = int(X.shape[0] * proportionTrain)
    X_train = X[:train_size, :, :]
    X_test = X[train_size: nn_size, :, :]
    y_train = y[:train_size]
    y_test = y[train_size: nn_size]
    res_train = res[:train_size]
    res_test = res[train_size: nn_size]
    return (X_train, y_train), (X_test, y_test), (res_train, res_test)

##--- 6.3 Split X and Y dataset into X_train X_test y_train y_test ---###
def get_train_test_return(dataX, dataY, nn_start, nn_size, proportionTrain, modeRC):
    Xs = np.array(dataX)
    ys = np.array(dataY)
    X = Xs[:,:]           # dataX with 'instrument', 'date', 'close', 'open', ...
    res = ys[:,2] * 1               # array with effective return over the horizon
    futClose = ys[:,3] * 1          # array with future close of the stock
    y = res * 1
    if modeRC == 'class':           # if classification => if return >=0 then buy, else let
        y[y >= 0] = 1   
        y[y < 0.1] = 0              # either 0 or -1 to emphasize the cost of error with custom loss function
        
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

##--- 6.3 Split X and Y dataset into X_train X_test y_train y_test ---###    
def get3D_train_test_return(dataX, dataY, nn_start, nn_size, proportionTrain, modeRC):
    X = dataX[:, :, :]    # dataX with 'instrument', 'date', 'close', 'open', ...
    res = dataY[:,2] * 1   # identifying the effective return achieve by the investment
    y = res * 1     # y = close price => regression on predicting future price
    if modeRC == 'class':           # if classification => if return >=0 then buy, else let
        y[y >= 0] = 1   
        y[y < 0.1] = -1
        y1 = y.reshape(len(y),1) * 1
        y2 = y.reshape(len(y),1) * 1
        y1[y1<1]=0
        y2[y2>0]=0
        y2[y2<0]=1
        y = np.append(y1, y2, axis = 1)
        nb_classes = 2
    else:
        nb_classes = 1
    if nn_start > 0:
        X = X[nn_start:, :, :]
        y = y[nn_start:,]
        res = res[nn_start:]
        
    if X.shape[0] > nn_size:           # apply the split on the lowest of lstmSize or X length
        train_size = int(nn_size * proportionTrain)
    else:
        train_size = int(X.shape[0] * proportionTrain)
    X_train = X[:train_size, :, :]
    X_test = X[train_size: nn_size, :, :]
    y_train = y[:train_size]
    y_test = y[train_size: nn_size]
    res_train = res[:train_size]
    res_test = res[train_size: nn_size]
    return (X_train, y_train), (X_test, y_test), (res_train, res_test, nb_classes)

##--- 6.4 MODE Multi-Class of train-test split for return based on yield for 1 EUR invested
def get3D_train_test_returnMC(dataX, dataY, nn_start, nn_size, proportionTrain, modeRC, trig_up=0, trig_down=0):
    X = dataX[:, :, :]    # dataX with 'instrument', 'date', 'close', 'open', ...
    res = dataY[:,2] * 1   # identifying the effective return achieve by the investment
    y = res * 1     # y = close price => regression on predicting future price
    if modeRC == 'class':           # if classification => if return >=0 then buy, else let
        nb_classes = 2
        if trig_up > 0:
            y[y >= trig_up] = 2
            y[(y>= 0) & (y<2)] = 1
            nb_classes += 1
        else:
            y[y >= 0] = 1   
            
        if trig_down < 0:
            y[y <= trig_down] = -2
            y[(y < 0) & (y>-2)] = -1
            nb_classes += 1
        else:
            y[y < 0.1] = -1
        y1 = y.reshape(len(y),1) * 1
        y2 = y.reshape(len(y),1) * 1
        y1[y1<1]=0
        y2[y2>0]=0
        y2[y2<-1]=2
        y2[y2<0]=1
        if trig_up > 0:
            y3 = y.reshape(len(y),1) * 1
            y3[y3 < 2] = 0
            y3[y3 > 1] = 1
            y1[y3 > 0] = 0
        if trig_down < 0:
            y4 = y.reshape(len(y),1) * 1
            y4[y4 > -2] = 0
            y4[y4 < -1] = 1
            y2[y4 > 0] = 0
        yt = np.append(y1, y2, axis = 1)
        if trig_up > 0:
            yt = np.append(yt, y3, axis = 1)
        if trig_down < 0:
            yt = np.append(yt, y4, axis = 1)
        y = yt * 1
    else:
        nb_classes = 1
    
    if nn_start > 0:
        X = X[nn_start:, :, :]
        y = y[nn_start:,]
        res = res[nn_start:]
        
    if X.shape[0] > nn_size:           # apply the split on the lowest of lstmSize or X length
        train_size = int(nn_size * proportionTrain)
    else:
        train_size = int(X.shape[0] * proportionTrain)
    X_train = X[:train_size, :, :]
    X_test = X[train_size: nn_size, :, :]
    y_train = y[:train_size]
    y_test = y[train_size: nn_size]
    res_train = res[:train_size]
    res_test = res[train_size: nn_size]
    return (X_train, y_train), (X_test, y_test), (res_train, res_test, nb_classes)

###--- 7. Scale X_train X_test on X_train ---###
## 7.1
def get_scaleX(X_train, X_test):
    train_mean = np.mean(X_train, axis = 0)
    train_std = np.std(X_train, axis = 0)
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    return (X_train, X_test), (train_mean, train_std)

def get_minmaxscaleX(X_train, X_test):
    train_mean = np.mean(X_train, axis = 0)
    train_minmax = np.amax(X_train, axis = 0) - np.amin(X_train, axis = 0)
    X_train = (X_train - train_mean) / train_minmax
    X_test = (X_test - train_mean) / train_minmax
    return (X_train, X_test), (train_mean, train_minmax)

## 7.2 Scaling for 3D dataset (LSTM, CNN, ResNet, ...)
def get3D_scaleX(X_train, X_test):
    train_mean = np.mean(X_train, axis = 0)
    train_mean = np.mean(train_mean, axis = 0)
    train_std = np.std(X_train, axis = 0)
    train_std = np.mean(train_std, axis = 0)
    X_train = X_train - train_mean.T
    X_train = X_train / train_std.T
    X_test = X_test - train_mean.T
    X_test = X_test / train_std.T
    return (X_train, X_test), (train_mean, train_std)

def get3D_minmaxscaleX(X_train, X_test):
    train_mean = np.mean(X_train, axis = 0)
    train_mean = np.mean(train_mean, axis = 0)
    train_minmax = np.amax(X_train, axis = 0) - np.amin(X_train, axis = 0)
    train_minmax = np.mean(train_minmax, axis = 0)
    X_train = X_train - train_mean.T
    X_train = X_train / train_minmax.T
    X_test = X_test - train_mean.T
    X_test = X_test / train_minmax.T
    return (X_train, X_test), (train_mean, train_minmax)


###--- 8. Split X matrix into X & Y with train & test for xgb_validation ---###

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

   