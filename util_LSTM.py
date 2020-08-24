# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 23:16:50 2020

Utility functions for financial analysis to prepare X and Y matrices
    1. LSTM_cleanXset(X_raw, trigger) => return X_clean
    2. create_dataset(dataX, dataY, seq_length, step) => return Xs, ys as np.array
    3. LSTM_train_test_size(dataX, dataY, lstmStart, lstmSize, proportionTrain, classRegr) => return (X_train, y_train), (X_test, y_test), (res_train, res_test)
    4. LSTM_clean_traintest(X_train, X_test, trigger) => return X_train, X_test
    5. LSTM_scaleX(X_train, X_test) => return Xs_train, Xs_test
    6.  => return 
    
@author: JDE65
"""
# ====  PART 0. Installing libraries ============

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc



###--- 1. Create sliding window and enrich dataX and dataY ---###

def LSTM_create_dataset(dataX, dataY, seq_length, step):
    Xs, ys = [], []
    for i in range(0, len(dataX) - seq_length, step):
        v = dataX.iloc[i:(i + seq_length)].values
        Xs.append(v)
        ys.append(dataY.iloc[i + seq_length])
    return np.array(Xs), np.array(ys)

def CONV_create_dataset(dataX, dataY, seq_length, step):
    Xs, ys = [], []
    for i in range(0, len(dataX) - seq_length, step):
        v = dataX.iloc[i:(i + seq_length)].values
        Xs.append(v)
        ys.append(dataY.iloc[i + seq_length])
    return Xs, ys

###--- 2. Create sliding window and enrich dataX and dataY ---###
def LSTM_train_test_price(dataX, dataY, nn_start, nn_size, proportionTrain):
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
    
def LSTM_train_test_return(dataX, dataY, nn_start, nn_size, proportionTrain, modeRC):
    X = dataX[:, :, :]    # dataX with 'instrument', 'date', 'close', 'open', ...
    res = dataY[:,2] * 1   # identifying the effective return achieve by the investment
    y = res * 1     # y = close price => regression on predicting future price
    
    if modeRC == 'class':           # if classification => if return >=0 then buy, else let
        y[y >= 0] = 1   
        y[y < 0.1] = -1
        
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


    
def CONV_train_test_size(dataX, dataY, lstmStart, lstmSize, proportionTrain, classRegr, truepos_enhancing, truepos_enhancing_trigger, falsepos_enhancing, falsepos_enhancing_trigger):
    X = dataX[:, :, :, :]    # dataX with 'instrument', 'date', 'close', 'open', ...
    res = dataY[:,2,:] * 1      # separating the effective return between train & test
    futClose = dataY[:,3,:] * 1
    y_Rclose = futClose * 1     # y = close price => regression on predicting future price
    y_Rret = res * 1            # y = future return => regression on predicting future return
    y_Cret = dataY[:,1, :] * 1           # Classification with base case non-invested : BASE = 0
    y_Cinv = dataY[:,1, :] * 1           # Classification with base case invested : BASE = 1 except if return is negative
    y_Cinv[y_Cinv >= 0] = 1
    y_Cinv[y_Cinv < .5] = 0
    
    if classRegr == 'Regr_close':
        y = y_Rclose * 1
    elif classRegr == 'Regr_ret':
        y = y_Rret * 1
        y[y > truepos_enhancing_trigger] *= truepos_enhancing   
        y[y < falsepos_enhancing_trigger] *= falsepos_enhancing
    elif classRegr == 'Class_ret':
        y = y_Cret *1
        y[y > truepos_enhancing_trigger] *= truepos_enhancing   
        y[y < falsepos_enhancing_trigger] *= falsepos_enhancing   
    else:
        y = y_Cinv *1        
        y[y > truepos_enhancing_trigger] *= truepos_enhancing   
        y[y < falsepos_enhancing_trigger] *= falsepos_enhancing  
    
    if lstmStart > 0:
        X = X[lstmStart:, :, :, :]
        y = y[lstmStart:,:]
        res = res[lstmStart:,:]
        
    if X.shape[0] > lstmSize:           # apply the split on the lowest of lstmSize or X length
        train_size = int(lstmSize * proportionTrain)
    else:
        train_size = int(X.shape[0] * proportionTrain)
    X_train = X[:train_size, :, :, :]
    X_test = X[train_size: lstmSize, :, :, :]

    y_train = y[:train_size, :]
    y_test = y[train_size: lstmSize, :]
    res_train = res[:train_size, :]
    res_test = res[train_size: lstmSize, :]
        
    return (X_train, y_train), (X_test, y_test), (res_train, res_test)


def get_LSTM_scaleX(X_train, X_test):
    train_mean = np.mean(X_train, axis = 0)
    train_mean = np.mean(train_mean, axis = 0)
    train_std = np.std(X_train, axis = 0)
    train_std = np.mean(train_std, axis = 0)
    X_train = X_train - train_mean.T
    X_train = X_train / train_std.T
    X_test = X_test - train_mean.T
    X_test = X_test / train_std.T
    return (X_train, X_test), (train_mean, train_std)

def get_Conv2D_scaleX(X_train, X_test):
    train_mean = np.mean(X_train, axis = 0)
    train_std = np.std(X_train, axis = 0)
    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    return (X_train, X_test), (train_mean, train_std)

    
"""
###--- 5. evaluate, predict and assess model
def LSTM_eval_decision(y_test, y_pred, X_test, classRegr, eval_trigger, train_mean = 0, train_std = 0, seq_length = 0):

    y_predict = y_pred * 1                  # create matrix with predicted results from NN
    y_testdec = y_pred * 1                  # initiate a matrix for the test optimal decision as target
    y_testdec[:,0] = y_test[:,]
    if (classRegr == 'Regr_close'):
        y_pred2 = (y_pred * train_std[seq_length-1,0]) + train_mean[seq_length-1,0]       # prediction of stock price unregularized
        y_test2 = (y_test * train_std[seq_length-1,0]) + train_mean[seq_length-1,0]       # effective stock price unregularized
        x_close = X_test[:,seq_length-1,0]
        y_predict = y_pred2 * 1
        y_testdec[:,0] = y_test2[:]
        x_close  = (x_close * train_std[seq_length-1,0]) + train_mean[seq_length-1,0]
        y_predict[(y_predict[:,0] / x_close - 1) > eval_trigger] = 1   # predicted price > close_price by expected trigger => buy
        y_predict[(y_predict[:,0] / x_close - 1) <= eval_trigger] = 0   # predicted price < close_price by expected trigger => NO buy
        y_testdec[(y_testdec[:,0] / x_close - 1) > eval_trigger] = 1    # future price > close_price by expected trigger => buy
        y_testdec[(y_testdec[:,0] / x_close - 1) <= eval_trigger] = 0    # future price > close_price by expected trigger => NO buy     
    elif classRegr == 'Regr_ret':
        y_predict[y_predict > eval_trigger] = 1            # y = future return => regression on predicting future price
        y_predict[y_predict <= eval_trigger] = 0
        y_testdec[y_testdec > eval_trigger] = 1
        y_testdec[y_testdec <= eval_trigger] = 0
    elif classRegr == 'Class_ret':
        y_predict[y_predict > eval_trigger] = 1              # Classification with base case non-invested : BASE = 0
        y_predict[y_predict < 1] = 0
        y_testdec[y_testdec > 0] = 1
        y_testdec[y_testdec <= 0] = 0
    else:
        y_predict[y_predict >= 1] = 1              # Classification with base case invested : BASE = 1
        y_predict[y_predict < 1] = 0
        y_testdec[y_testdec > 0] = 1
        y_testdec[y_testdec < 1] = 0
    
    confmat = confusion_matrix(y_testdec, y_predict)
    
    return y_testdec, y_predict, confmat
"""
