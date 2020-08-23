# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:43:36 2020
Simple DENSE NN model for timeseries regression / classification for asset return
Divided into .A - input management and .B - model management
Inputs required in Part 0.A and Part 2.B 
@author: JDE65
"""

# ====  PART 0. Installing libraries ============

import numpy as np
import pandas as pd
import sqlite3 as sq
import time
from itertools import chain
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.layers import Dropout, Activation, Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix

from util_enrich import *
from util_prepa import *
from util_model import *

start_time = time.time()
rcParams['figure.figsize'] = 14, 8
### ====   PART 0.A Defining hyeprparameters & parameters  =  INPUT REQUIRED ============
## SQL parameters
dbInput = 'CAC40stock.db'           ### Database with input data
dbList = "TRlistCAC40"              ### table with list of stocks
ric = "BOUY"                        ### RIC code of the stock
dbOutput = 'TradLSTMCAC40.db'       ### Database for saving output
saveX = "tradX"                     ### Table for saving X output in dbOutput
saveY = "tradY"                     ### Table for saving Y output in dbOutput

## Dataset parameters
horiz = 1                          ### time horizon of the investment in days ###
eval_trigger = 0.005
arima_p = 0
arima_d = 0
arima_q = 0
trigger_drop = 0.1                      ### percentage of nan in a column of X that causes deletion of the column
drop_rows = 50                      ### Number of unrelevant rows given technical indicators computation
nn_start = 0                    ### initial value of X and Y matrices out of the total dataset
nn_size = 2000                     ### length of the X & Y matrices starting from lstmStart index
proportionTrain = 0.9875
trigger = 0.1                      ### percentage of nan in a column of X that causes deletion of the column

modeRC = 'class'                 ### 'class' = classification  |  'regr' = regression  
class_trig = 0.45                   ### trigger for investment decision for classification
regr_trig = 0                  ### trigger for investment decision for regression
if modeRC == 'class':
    eval_trigger = class_trig
else:
    eval_trigger = regr_trig

X_plot = 0                          ### 1 for plot close price  /  0 for no plot

### ====   PART 1.A Connecting to SQL DB and loading lists ============
dataX, dataY = get_model_data(dbInput, dbList, ric, horiz, drop_rows)
dataX = get_technical_indicators(dataX)
#dataX = get_arima(dataX, arima_p, arima_q, arima_d)
#dataX = get_correlated_assets(dataX)
dataX = get_droprows(dataX, drop_rows)
dataY = get_droprows(dataY, drop_rows)
dataX = get_cleandupli_dataset(dataX)                       # prevent duplicated columns
dataX = get_model_cleanXset(dataX, trigger_drop)                 # Clean X matrix for insufficient data


ys = np.array(dataY)
res = ys[:,2] * 1               # array with effective return over the horizon
futClose = ys[:,3] * 1          # array with future close of the stock
all_result_BH = 0
all_result_pred = 0
all_result_BT = 0
all_deal_BH = 0
all_deal_pred = 0
all_deal_BT = 0
### === 
nbsteps = int((len(dataX)-nn_size) / (nn_size * (1 - proportionTrain)))
for nbstep in range(nbsteps - 1):
    nn_start = int(nbstep * (nn_size * (1 - proportionTrain)))
    
    (X_train, y_train), (X_test, y_test), (res_train, res_test) = get_train_test_return(dataX, dataY, nn_start, nn_size, proportionTrain, modeRC)
    (X_train, X_test), (train_mean, train_std) = get_model_scaleX(X_train, X_test)

### ====   PART 2.B Input & define Model  =  INPUT REQUIRED ============
## Model & Hyper-parameters
    validation_split = 0.05
    model = keras.Sequential()
    dropout = 0.1
    optimizer = 'adam'                      ### Optimizer of the compiled model
    learning = 0.001
    loss = custom_return_loss_function
    # loss = 'mean_squared_error'
    verbose = 0                         ### 0 = hidden computation  //  1 = computation printed
    batch_size = 128
    epochs = 50
    layer_1 = 512
    layer_2 = 1024

# available layers
    layer_drop = keras.layers.Dropout(rate = dropout)
    layer_dense1 = Dense(units= layer_1, activation = 'relu')
    layer_dense2 = Dense(units= layer_2, activation = 'relu')
    layer_output = Dense(units = 1)
    
# Model architecture 
    model.add(layer_dense2)
    model.add(layer_drop)
    model.add(layer_dense1)
    model.add(layer_output)
    model_arch = 'D1-256+Drop+D128+Out'
                    
### ====   PART 3.B Plot data X (optional)   ============
    if (X_plot == 1):
        plot_cols = [3, 15, 17]
        plot_features = dataX[plot_cols][nn_start:(nn_start + nn_size)]
        plot_features = plot_features.reset_index(drop = True)
        _ = plot_features.plot(subplots=False)
        print(dataX.describe().transpose())
    print('Time preparing data = ',f'Time: {time.time() - start_time}')
    
### ====   PART 4.B Compile and Train model + predict   ============
    model, history = model_compile_train(model, loss, optimizer, X_train, y_train, epochs, batch_size, validation_split, verbose)
    eval_train, eval_test, y_pred = model_predict(model, history, X_train, y_train, X_test, y_test)

    plt.plot(y_test)
    plt.plot(y_pred)
    plt.title('Return : real vs predicted')
    plt.ylabel('Return')
    plt.xlabel('Days')
    plt.legend(['Real return', 'Predicted return'], loc='upper left')
    plt.show()

## compute investment proposition and its accuracy
    y_testdec, y_predict, confmat = model_eval_return(res_test, y_test, y_pred, eval_trigger)
    if confmat.shape[0] == 1:
        precis = 1 * 0
        recall = 1 * 0
        F1 = 1 * 0
    else:
        precis, recall, F1 = get_F1(confmat)        # compute precision, recall and F1 of the regression / classification           

## compute the financial result of the suggested investment decisions
    (nbDeals_bhst, nbDeals_test, nbDeals_pred), (bhst_result, test_result, pred_result) = model_eval_result(y_testdec, y_predict, res_test)
    eval_out_net = np.array([precis, recall, F1, pred_result, nbDeals_pred, bhst_result, test_result, all_result_BH, all_result_pred, all_result_BT])
    eval_out_net = eval_out_net.reshape(10, 1)
    if nbstep == 0:
        eval_output_net = eval_out_net * 1
        eval_output_net = eval_output_net.reshape(10, 1)
    else:
        eval_output_net = np.append(eval_output_net, eval_out_net, axis = 1)
    
### ====   PART 5.B Plot & Output   ============

    print('Model precision   |    recall       |   F1 ratio')
    print('      ',round(precis,5),'   |   ',round(recall, 5),'     |   ', round(F1, 5))
    print('Analysed result  : ',
          'B&H strategy |     ', 
          'Prediction   |     ', 
          'Optimal test')
    print('total result    : ',
          round(bhst_result, 5), 
          '            ',
          round(pred_result, 5),
          '           ',
          round(test_result, 5))
    print('Number of deals : ',
          nbDeals_bhst,
          '                 ',
          nbDeals_pred, 
          '              ',
          nbDeals_test)
    print('Average result  : ',
          round(bhst_result / nbDeals_bhst * 100, 4),'%'
          '            ',
          round(pred_result / nbDeals_pred * 100, 4),'%' 
          '           ',
          round(test_result / nbDeals_test * 100, 4),'%')
    
    print('Time for this run = ',f'Time: {time.time() - start_time}')
    all_result_BH += bhst_result
    all_result_pred += pred_result
    all_result_BT += test_result
    all_deal_BH += nbDeals_bhst
    all_deal_pred += nbDeals_pred
    all_deal_BT += nbDeals_test
# Plot general summary
plt.plot(eval_output_net[0, :])
plt.plot(eval_output_net[1, :])
plt.plot(eval_output_net[2, :])
plt.title('model average accuracy through time')
plt.ylabel('Precision & recall')
plt.xlabel('Time')
plt.legend(['precision', 'recall', 'F1'], loc='upper left')
plt.show()

plt.plot(eval_output_net[3, :])
plt.plot(eval_output_net[5, :])
plt.plot(eval_output_net[6, :])
plt.title('model total result accuracy through time')
plt.ylabel('Total result')
plt.xlabel('Time')
plt.legend(['model prediction', 'Buy & Hold', 'Back trading'], loc='upper left')
plt.show()

plt.plot(eval_output_net[8, :])
plt.plot(eval_output_net[7, :])
plt.plot(eval_output_net[9, :])
plt.title('model Cumulative result accuracy through time')
plt.ylabel('Cumulative Total Result')
plt.xlabel('Time')
plt.legend(['model prediction', 'Buy & Hold', 'Back trading'], loc='upper left')
plt.show()

print('Time for the TOTAL run = ',f'Time: {time.time() - start_time}')








