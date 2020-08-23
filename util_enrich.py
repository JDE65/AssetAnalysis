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
import sqlite3 as sq
import matplotlib.pyplot as plt
from itertools import chain
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf

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
    dataset['mav05'] = pd.Series(mav05, index = dataset.index)
    dataset['mav20'] = pd.Series(mav20, index = dataset.index)
    dataset['expmav12'] = pd.Series(expmav12, index = dataset.index)
    dataset['expmav26'] = pd.Series(expmav26, index = dataset.index)
    dataset['expmav50'] = pd.Series(expmav50, index = dataset.index)
    dataset['macd'] = pd.Series(macd, index = dataset.index)
    dataset['BollUp20'] = pd.Series(bollUpperband, index = dataset.index)
    dataset['BollDown20'] = pd.Series(bollLowerband, index = dataset.index)
    dataset['std20'] = pd.Series(rstd20, index = dataset.index)
    dataset['CCI14'] = pd.Series(cci, index = dataset.index)
    dataset['EVM14'] = pd.Series(evm_ma, index = dataset.index)
    dataset['ATR10'] = pd.Series(atr10, index = dataset.index)
    dataset['+DX5'] = pd.Series(posdmi5, index = dataset.index)
    dataset['-DX5'] = pd.Series(negdmi5, index = dataset.index)
    dataset['ADX5'] = pd.Series(adx5, index = dataset.index)
    dataset['ADX14'] = pd.Series(adx14, index = dataset.index)
    dataset['RSI05'] = pd.Series(rsi5, index = dataset.index)
    dataset['RSI14'] = pd.Series(rsi14, index = dataset.index)
    dataset['Momentum1'] = pd.Series(momentum1, index = dataset.index)
    
    return dataset

###--- 2. Add prices of same index assets to a dataset ---###
def get_stocks_sameindex(dataset, dbInput):
    connect2 = sq.connect(dbInput)        # connect to the DB with the stock price information
    cursor2 = connect2.cursor()
    tableSelect = "Select RIC FRM " + dbInput
    cursor2.execute(tableSelect)      # Select table with Indexes
    outp2 = cursor2.fetchall()
    colname = list(chain(*outp2))
    colIndexname = list(chain(*outp2))
    cursor2 = connect2.cursor()
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

###--- 5. Add Fourier  + summary ---###
def get_fourier(dataset, fourier1):
    return dataset


###--- 5. Drop unnecessary rows into a dataset + summary ---###
def get_dropdrows(dataset, num_rows):
    dataset = dataset.drop(dataset.index[0:num_rows])
    return dataset


###--- 6. To be developped ---###

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

   