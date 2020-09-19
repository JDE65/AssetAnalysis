# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 19:54:21 2020
Utility functions for financial analysis to prepare X and Y matrices
    1. get_technical_indicators(dataset) => return dataset
    2. get_stocks_sameindex(dataset) => return dataset
    3. def get_correlated_assets(dataset) => return dataset
    4. get_arima(dataset, p, d, q) => return TO BE DEVELOPPED
    5. get_fourier(dataset, fourier1) => return TO BE DEVELOPPED
    6. plot_technical_indicators(dataset, last_days) => TO BE DONE
    
    
@author: JDE65
"""
import pandas as pd
import numpy as np
import sqlite3 as sq
import matplotlib.pyplot as plt
from itertools import chain
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf
import pmdarima
#from util_prepa import *

###--- 0. Load Data for any NN model
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

def get_model_Y(dataset, horizon, triggerUp1 = 0, triggerLoss = 0, triggerUp2 = 0):
    dataY = pd.DataFrame(columns = ['date', 'yval', 'returnF', 'futClose'])
    dataY['date'] = dataset['date']
    dataY['yval'] = dataY['yval'].astype(float)                                # initialize y value
    dataY['returnF'] = dataY['returnF'].astype(float)
    dataY['futClose'] = dataY['futClose'].astype(float)
    #dataY['yval'] = 0                                # initialize y value
    #dataY['returnF'] = 0                             #reinitialize returnF (future return value)
        
    for i in range(len(dataset)):
        if (i < (len(dataset) - horizon)):                  # if row above the last number of rows for which indicators are not meaningfull
            futClose = dataset.at[i + horizon, 'close']    
            perfFut = (futClose / dataset.at[i, 'close']) - 1    # invest at close of day
            #perfFut = (futClose / dataset.at[i + 1, 'open']) - 1   #Invest next morning at opening
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
    return dataY

###--- 1. Add technical indicators to a dataset ---###
def get_technical_indicators(dataset):
    # moving average
    mav05 = dataset['close'].rolling(window=5).mean()
    mav20 = dataset['close'].rolling(window=20).mean()
    expmav12 = dataset['close'].ewm(span = 12, adjust=False).mean()          # Using Pandas to calculate a 5 or 20-days span EMA. adjust=False specifies that we are interested in the recursive calculation mode.
    expmav26 = dataset['close'].ewm(span = 26, adjust=False).mean()
    expmav50 = dataset['close'].ewm(span = 50, adjust=False).mean()

    # MACD   -   Moving Average Convergence Divergence
    macd = (expmav26 - expmav12)                                  #computing moving average convergence divergence

    # Bollinger Bands
    rstd20 = dataset['close'].rolling(window=20).std()
    bollUpperband = mav20 + 2 * rstd20
    bollLowerband = mav20 - 2 * rstd20

    # CCI - Commodity Channel Index
    techpr = (dataset['close'] + dataset['high'] + dataset['low']) / 3
    cci = (techpr - techpr.rolling(14).mean())/(techpr.rolling(14).std())

    # EMV - Ease of Movement 
    dm = ((dataset['high'] + dataset['low']) / 2) - ((dataset['high'].shift(1) + dataset['low'].shift(1)) / 2)
    br = (dataset['volume'] / 1000000) / ((dataset['high'] - dataset['low']))
    evm = dm / br
    evm_ma = evm.rolling(14).mean()

    # ATR - Average True Range 
    tr1 = (dataset['high'] - dataset['low'])
    tr2 = abs(dataset['high'] - dataset['close'].shift(1))
    tr3 = abs(dataset['low'] - dataset['close'].shift(1))
    trRange = tr1
    trRange[tr2>tr1] = tr2[tr2 > tr1]
    trRange[tr3>trRange] = tr3[tr3 > trRange]
    atr10 = trRange
    atr10 = (atr10.shift(-1) * (10 - 1) + trRange)/10           # a corriger pour le lag

    # ADX  -  Average Directionnal Index (relies on ATR)
    tr1 = dataset['high'] - dataset['high'].shift(1)
    tr2 = (dataset['low'].shift(1) - dataset['low'])
    dirmovup = 0 * abs(tr1)
    dirmovd = 0 * abs(tr1)
    dirmovup[tr1>tr2] = tr1[tr1>tr2]
    dirmovup[dirmovup<0] = dirmovup[dirmovup<0] * 0
    dirmovd[tr2>tr1] = tr2[tr2>tr1]
    dirmovd[dirmovd<0] = dirmovd[dirmovd<0] * 0
    dirmovup5 = dirmovup.rolling(5).mean()
    dirmovd5 = dirmovd.rolling(5).mean()
    trRange5 = trRange.rolling(5).mean()
    posdmi5 = dirmovup5 / trRange5          # positive Directional movement Indicator
    negdmi5 = dirmovd5 / trRange5          # negative Directional movement Indicator
    dx = (posdmi5 - negdmi5) / (posdmi5 + negdmi5)
    adx5 = dx.rolling(5).mean()             # average directionnal index - magnitude of the expected trend both ways
    dirmovup14 = dirmovup.rolling(14).mean()
    dirmovd14 = dirmovd.rolling(14).mean()
    trRange14 = trRange.rolling(5).mean()
    posdmi14 = dirmovup14 / trRange14          # positive Directional movement Indicator
    negdmi14 = dirmovd14 / trRange14          # negative Directional movement Indicator
    dx14 = (posdmi14 - negdmi14) / (posdmi14 + negdmi14)
    adx14 = dx14.rolling(14).mean()             # average directionnal index - magnitude of the expected trend both ways

    # RSI   -   Relative Strength Index
    diff = dataset['close'] - dataset['close'].shift(1)
    upd = 0 * abs(diff)
    downd = 0 * abs(diff)
    upd[diff>0] = diff[diff>0]
    downd[diff<0] = -diff[diff<0]
    rs5 = upd.rolling(5).mean() / downd.rolling(5).mean()
    rsi5 =  100 - (100 / (1 + rs5))
    rs14 = upd.rolling(14).mean() / downd.rolling(14).mean()
    rsi14 =  100 - (100 / (1 + rs14))

    # Create Momentum
    momentum1 = dataset['close'] / dataset['close'].shift(1)
    
    # = Adding parametrs to dataframe =
    #dataset['mav05'] = pd.Series(mav05, index = dataset.index)
    #dataset['mav20'] = pd.Series(mav20, index = dataset.index)
    dataset['expmav12'] = pd.Series(expmav12, index = dataset.index)
    dataset['expmav26'] = pd.Series(expmav26, index = dataset.index)
    dataset['expmav50'] = pd.Series(expmav50, index = dataset.index)
    dataset['macd'] = pd.Series(macd, index = dataset.index)
    dataset['BollUp20'] = pd.Series(bollUpperband, index = dataset.index)
    dataset['BollDown20'] = pd.Series(bollLowerband, index = dataset.index)
    #dataset['std20'] = pd.Series(rstd20, index = dataset.index)
    dataset['CCI14'] = pd.Series(cci, index = dataset.index)
    #dataset['EVM14'] = pd.Series(evm_ma, index = dataset.index)
    dataset['ATR10'] = pd.Series(atr10, index = dataset.index)
    #dataset['+DX5'] = pd.Series(posdmi5, index = dataset.index)
    #dataset['-DX5'] = pd.Series(negdmi5, index = dataset.index)
    dataset['ADX5'] = pd.Series(adx5, index = dataset.index)
    dataset['ADX14'] = pd.Series(adx14, index = dataset.index)
    dataset['RSI05'] = pd.Series(rsi5, index = dataset.index)
    dataset['RSI14'] = pd.Series(rsi14, index = dataset.index)
    #dataset['Momentum1'] = pd.Series(momentum1, index = dataset.index)
    return dataset

###--- 2. Add prices of same index assets to a dataset ---###
def get_stocks_sameindex(dataset, dbInput, dbList, ric, rics, horiz):
    dataset = dataset.set_index('date')
    for nric in rics:        
        if nric != ric:
            connect = sq.connect(dbInput)
            cursor = connect.cursor()
            dataRIC = 'data' + nric
            tableforRIC = "SELECT * FROM "
            dbRIC = tableforRIC + dataRIC
            cursor.execute(dbRIC)
            outp = cursor.fetchall()
            dataTrad = pd.DataFrame(outp, columns =['index', 'reindex', 'instrument2', 'date', 'close2', 'open', 'high', 'low', 'dvt', 'volume', 'dtr', '1wktr', '1mthtr', '3mthtr'])
            dataTrad.drop(dataTrad[pd.isnull(dataTrad["close2"])].index, inplace=True)           # clean DB by dropping all rows where closing price is null
            dataTrad['date'] = dataTrad['date'].str[:4] + dataTrad['date'].str[5:7] + dataTrad['date'].str[8:10]       #convert long string in 8 character string for date formating
            dataTrad['date'] = dataTrad['date'].astype(int)
            colname = nric + 'close'
            dataTrad = dataTrad.drop(columns=['index','reindex', 'open','high', 'low', 'dvt', 'volume', 'dtr', '1wktr', '1mthtr', '3mthtr'])
            dataset = dataset.join(dataTrad.set_index('date')).fillna(method='ffill')
            dataset.rename(columns={'close2':colname}, inplace=True)
            dataset = dataset.drop(columns=['instrument2'])
            dataset = get_cleandupli_dataset(dataset)
    return dataset
    
###--- 3. Add correlated assets to a dataset ---###
def get_correlated_assets(dataset):
    connect2 = sq.connect('CorrelatedAssets.db')        # connect to the DB with the stock price information
    cursor2 = connect2.cursor()
    cursor2.execute("SELECT RIC FROM CorrelatedAssets")      # Select table with Indexes
    outp2 = cursor2.fetchall()
    colname = list(chain(*outp2))
    cursor2 = connect2.cursor()
    cursor2.execute("SELECT RIC FROM CorrAssIndex")          # Select table with Indexes
    outp2 = cursor2.fetchall()
    colIndexname = list(chain(*outp2))
    cursor2 = connect2.cursor()
    cursor2.execute("SELECT RIC FROM CorrAssIBOR")           # Select table with Indexes
    outp2 = cursor2.fetchall()
    colIBORname = list(chain(*outp2))
    cursor2 = connect2.cursor()
    cursor2.execute("SELECT RIC FROM CorrAssFXO")            # Select table with Indexes
    outp2 = cursor2.fetchall()
    colFXOname = list(chain(*outp2))
    dataCA = pd.DataFrame(columns = colname)            # initialize DataFrame with columns for all correlated assets
    dataCA.insert(0, 'date', dataCA.mean(1))            # insert a column with the dates
    dataCA['date'] = dataset['date']                   # initiate a Dataframe of the length of X

    ## Enrich dataCA with the indexes
    cursor2.execute("SELECT * FROM dataCorrIndex")          # Select table with Indexes
    outp2 = cursor2.fetchall()
    dataCorrIndex = pd.DataFrame(outp2, columns =['index', 'instrument', 'date', 'close'])
    dataCorrIndex['date'] = dataCorrIndex['date']. str[:4] + dataCorrIndex['date']. str[5:7] +dataCorrIndex['date']. str[8:10]       #convert long string in 10 character string for date formating
    dataCorrIndex['date'] = dataCorrIndex['date'].astype(int)       # convert date into integer format
    dataCorrIndex = dataCorrIndex.drop('index', 1)                  # clean the data with correlated indexes
    dataCorrIndex = dataCorrIndex.drop_duplicates()                 # remove possible duplicates

    for instr in colIndexname:                                      #populate data CA with indexes values per date
        tempI = dataCorrIndex[dataCorrIndex['instrument']==instr]
        tempI = tempI.drop('instrument', 1)
        data2 = pd.DataFrame(columns = ['date'])
        data2['date'] = dataset['date']
        data2 = data2.merge(tempI, on= 'date', how = 'left')
        dataCA[instr] = data2['close']

    ## Enrich dataCA with the IBOR    
    cursor2.execute("SELECT * FROM dataCorrIBOR")           # Select table with IBOR
    outp2 = cursor2.fetchall()
    dataCorrIBOR = pd.DataFrame(outp2, columns =['index', 'instrument', 'date', 'close'])
    dataCorrIBOR['date'] = dataCorrIBOR['date']. str[:4] + dataCorrIBOR['date']. str[5:7] + dataCorrIBOR['date']. str[8:10]      #convert long string in 10 character string for date formating
    dataCorrIBOR['date'] = dataCorrIBOR['date'].astype(int)       # convert date into integer format
    dataCorrIBOR = dataCorrIBOR.drop('index', 1)                  # clean the data with correlated indexes
    dataCorrIBOR = dataCorrIBOR.drop_duplicates()                 # remove possible duplicates

    for instr in colIBORname:                                      #populate data CA with indexes values per date
        tempI = dataCorrIBOR[dataCorrIBOR['instrument']==instr]
        tempI = tempI.drop('instrument', 1)
        data2 = pd.DataFrame(columns = ['date'])
        data2['date'] = dataset['date']
        data2 = data2.merge(tempI, on= 'date', how = 'left')
        dataCA[instr] = data2['close']

    ## Enrich dataCA with the FXO
    cursor2.execute("SELECT * FROM dataCorrFXO")           # Select table with FXO - FX and others
    outp2 = cursor2.fetchall()
    dataCorrFXO = pd.DataFrame(outp2, columns =['index', 'instrument', 'date', 'close'])
    dataCorrFXO['date'] = dataCorrFXO['date']. str[:4] + dataCorrFXO['date']. str[5:7] + dataCorrFXO['date']. str[8:10]        #convert long string in 10 character string for date formating
    dataCorrFXO['date'] = dataCorrFXO['date'].astype(int)       # convert date into integer format
    dataCorrFXO = dataCorrFXO.drop('index', 1)                  # clean the data with correlated indexes
    dataCorrFXO = dataCorrFXO.drop_duplicates()                 # remove possible duplicates

    for instr in colFXOname:                                      #populate data CA with indexes values per date
        tempI = dataCorrFXO[dataCorrFXO['instrument']==instr]
        tempI = tempI.drop('instrument', 1)
        data2 = pd.DataFrame(columns = ['date'])
        data2['date'] = dataset['date']
        data2 = data2.merge(tempI, on= 'date', how = 'left')
        dataCA[instr] = data2['close']

    dataset = dataset.merge(dataCA, on= 'date', how = 'left')
    dataset = dataset.drop_duplicates()

    return dataset

###--- 4. Add arima series to a dataset + summary ---###
def get_arima(dataset, p, d, q):
    series = dataset['close']
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit(disp=0)
    summary = model_fit.summary()
    residuals = pd.DataFrame(model_fit.resid)
    return model_fit, summary, residuals

def get_auto_arima(dataset):
    arima_model = auto_arima(dataset, start_p=1, start_q=1, max_p=5, max_q=5,
                             start_P=0, start_Q=0, max_P=5, max_Q=5, m=7, 
                             seasonal=False, trace=True, d=1, D=1, error_action='warn',
                             supress_warnings=True, stepwise=True, random_state=20, n_fits=30)
    arima_pred = arima_model.predict(n_periods=44)
    dataset['Arima'] = pd.Series(arima_pred, index = dataset.index)
    return dataset

###--- 5. Add Fourier  + summary ---###
def compute_fourier(dataset):               # Compute Fourier transform for a given dataset
    close_fft = np.fft.fft(np.asarray(dataset['close'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute']=fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle']=fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9, 100]:
        fft_list_m10 = np.copy(fft_list); fft_list_m10[num_:-num_]=0
        fourier_name = 'fourier'+str(num_)
        dataset[fourier_name] = pd.DataFrame(np.fft.ifft(fft_list_m10)).apply(lambda x: np.abs(x))
    return dataset

def get_fourier(dataset, nn_start, nn_train, nn_test):
    dataset = compute_fourier(dataset)
    dataX_train = dataset.iloc[0 : (nn_start + nn_train), :] * 1
    dataX_train = compute_fourier(dataX_train)
    dataset.at[0:(nn_start + nn_train-1)] = dataX_train[:]
    for ii in range(nn_test):
        dataX2 = dataset.iloc[0 : (nn_train + ii + 1), :] * 1
        dataX2 = compute_fourier(dataX2)
        dataset.at[nn_train + ii] = dataX2.iloc[nn_train + ii] * 1
    return dataset

def print_fourier(dataset):
    close_fft = np.fft.fft(np.asarray(dataset['close'].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute']=fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle']=fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())
    plt.figure(figsize=(14,7), dpi=100)
    for num_ in [3, 6, 9, 100]:
        fft_list_m10 = np.copy(fft_list); fft_list_m10[num_:-num_]=0
        plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
        fourier_name = 'fourier'+str(num_)
        dataset[fourier_name] = pd.DataFrame(np.fft.ifft(fft_list_m10)).apply(lambda x: np.abs(x))
    plt.plot(dataset['close'],  label='Real')
    plt.xlabel('Days')
    plt.ylabel('USD')
    plt.title('Figure 3: Amazon (close) stock prices & Fourier transforms')
    plt.legend()
    plt.show()
    return dataset


   