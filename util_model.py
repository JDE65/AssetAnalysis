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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import tensorflow as tf

import tensorflow.keras
from tensorflow.keras import backend as K




###--- 2. Custom loss functions for training ---###
def custom_return_loss_function(y_actual, y_predicted):
    diff_y = K.square(y_actual- np.squeeze(y_predicted))
    bool_idx = K.greater(diff_y, K.square(y_actual))
    loss1 = tf.math.scalar_mul(2, diff_y)
    y_diff = K.switch(bool_idx, loss1, diff_y)
    #diff_y[(y_actual > 0) & (np.squeeze(y_predicted) < 0)] *= 2
    #diff_y[(y_actual < 0) & (np.squeeze(y_predicted) > 0)] *= 3
    #diff_y = K.constant(diff_y)
    custom_return_loss_value = K.mean(y_diff)
    return custom_return_loss_value


def custom_price_loss_function(y_actual, y_predicted): ### To be IMPROVED
    diff_y = np.abs(y_actual- np.squeeze(y_predicted))
    diff_y[(y_actual > 0) & (np.squeeze(y_predicted) < 0)] *= 2
    diff_y[(y_actual < 0) & (np.squeeze(y_predicted) > 0)] *= 3
    custom_price_loss_value = K.mean(diff_y)
    return custom_price_loss_value

###--- 3. Compile model for training ---###
def model_compile_train(model, loss, optimizer, X_train, y_train, epochs, batch_size, validation_split, verbose = 0):
    model.compile(loss = loss, optimizer = optimizer)
    history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = validation_split, verbose = verbose)
    return model, history


###--- 4. evaluate, predict and assess 
def model_predict(model, history, X_train, y_train, X_test, y_test):
    eval_train = model.evaluate(X_train, y_train)
    eval_test = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    return eval_train, eval_test, y_pred


###--- 4. evaluate, predict and assess 
def model_eval_price(X_ref, y_test, y_pred, eval_trigger):
    y_testdec = (y_test - X_ref) / X_ref
    y_predict = (y_pred[:,0] - X_ref) / X_ref
    y_predict[y_predict[:,] > eval_trigger] = 1   # predicted price > close_price by expected trigger => buy
    y_predict[y_predict[:,] <= eval_trigger] = 0   # predicted price < close_price by expected trigger => NO buy
    y_testdec[y_testdec[:,] > eval_trigger] = 1    # future price > close_price by expected trigger => buy
    y_testdec[y_testdec[:,] <= eval_trigger] = 0
    confmat = confusion_matrix(y_testdec, y_predict)
    return y_testdec, y_predict, confmat

def model_eval_return(res, y_test, y_pred, eval_trigger):
    y_testdec = y_test * 1
    y_predict = y_pred[:,0] * 1
    y_predict[y_predict[:,] >= eval_trigger] = 1   # predicted price > close_price by expected trigger => buy
    y_predict[y_predict[:,] < eval_trigger] = 0   # predicted price < close_price by expected trigger => NO buy
    y_testdec[y_testdec[:,] >= eval_trigger] = 1    # future price > close_price by expected trigger => buy
    y_testdec[y_testdec[:,] < eval_trigger] = 0
            
    confmat = confusion_matrix(y_testdec, y_predict)
    return y_testdec, y_predict, confmat


###--- 5. Analyse prediction F1
def get_F1(confmat):    
    true_neg = confmat[0,0]
    true_pos = confmat[1,1]
    false_neg = confmat[1,0]
    false_pos = confmat[0,1]
    precis = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    F1 = 2 * precis * recall / (precis + recall)
    
    return precis, recall, F1

def model_eval_result(y_testdec, y_predict, res_test):
    test_result, pred_result = [], []
    test_res = np.multiply(y_testdec[:,], res_test[:])
    pred_res = np.multiply(res_test[:], y_predict[:,])
    test_result = test_res.sum()
    pred_result = pred_res.sum()
    bhst_result = res_test.sum()
    nbDeals_test = y_testdec[y_testdec == 1].sum()
    nbDeals_pred = y_predict[y_predict == 1].sum()
    nbDeals_bhst = len(y_testdec)    

    
    return (nbDeals_bhst, nbDeals_test, nbDeals_pred), (bhst_result, test_result, pred_result)












###--- 3. Cleanup X for missing values ---###

def LSTM_clean_traintest(X_train, X_test, trigger):
    nanc = X_train.isna().sum()
    for i in range(len(nanc)):
        ananc = nanc.iloc[i]/len(X_train)
        if ananc > trigger:
            X_train = X_train.drop(columns = i)
            X_test = X_test.drop(columns = i)
    X_train = X_train.T.reset_index(drop = True).T
    X_test = X_test.T.reset_index(drop = True).T
    return X_train, X_test





###--- 5. Analyse NN efficiency
def get_stat_NN(y_train, y_test, y_pred, y_testdec) :
    #res_train = confusion_matrix(y_train, y_pred)
    res_test =  confusion_matrix(y_test, y_testdec)
    TPt = np.sum(res_test[1:, 1:]) # + res_train[2, 2] + res_train[1, 2] + res_train[2, 1]   # true positive includes exact prediction as well as event happening but sooner or later
    FPt = np.sum(res_test[0, 1:])   # event announced but not happening
    FNt = np.sum(res_test[1:, 0])     # event unannounced
        
    if ((TPt + FPt)!= 0):
        prect = TPt / (TPt + FPt)
        recallt = TPt / (TPt + FNt)
        F1t = 2 * prect * recallt / (prect + recallt)
        testAcc = (res_test[0, 0] + TPt) / np.sum(res_test)
        testAccDet = np.trace(res_test) / np.sum(res_test)
    else:
        prect = 0
        recallt = 0
        F1t = 0
        testAcc = (res_test[0, 0] + TPt) / np.sum(res_test)
        testAccDet = np.trace(res_test) / np.sum(res_test)

    return (prect, recallt, F1t), (testAcc, testAccDet)