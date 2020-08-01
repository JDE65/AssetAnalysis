# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:59:03 2020

@author: JDE65
"""
# ====   Installing libraries ============
import numpy as np
import pandas as pd
import sqlite3 as sq

#↔ import numpy as np
# from sqlalchemy import create_engine

# ====   Defining parameters ============
horiz = 15       #time horizon of the investment
dbSave = "trad15dCAC40r10"
#ricIndex = 'MT.AS'
mlp = 60  # maximum length of technical indicators (mav, bollband, ...)
returnA = .1   # first level of gain
returnB = .99  # second level of gain
returnC = 0.99   # third level of gain
returnD = -.99  #level of identified loss

# ====   Connecting to SQL DB and loading lists ============
connect = sq.connect('tradingCAC40.db')        # connect to the DB FinancialDB
cursor = connect.cursor()
tradeStock = pd.read_sql_query("SELECT * FROM TRdataTradCAC40", connect)
dataTrad = tradeStock.rename(columns={"Price Open": "open", "Price Close": "close", 
                                      "Price High": "high", "Price Low": "low", "Volume": "volume", "index": "reindex" })
dataTrad.drop(dataTrad[pd.isnull(dataTrad["close"])].index, inplace=True)            # clean DB by dropping all rows where closing price is null
dataTrad.drop(dataTrad[pd.isnull(dataTrad["open"])].index, inplace=True)            # clean DB by dropping all rows where closing price is null
#dataTrad.drop(dataTrad[(dataTrad["Instrument"] == ricIndex)].index, inplace=True)                                                     # eliminate the data of the index

# ====   Computing technical parameters ============
print('Enriching DB')
# moving average
mav05 = dataTrad['close'].rolling(window=5).mean()
mav20 = dataTrad['close'].rolling(window=20).mean()
expmav12 = dataTrad['close'].ewm(span = 12, adjust=False).mean()          # Using Pandas to calculate a 5 or 20-days span EMA. adjust=False specifies that we are interested in the recursive calculation mode.
expmav26 = dataTrad['close'].ewm(span = 26, adjust=False).mean()
expmav50 = dataTrad['close'].ewm(span = 50, adjust=False).mean()

# MACD
macd = (expmav26 - expmav12)                                  #computing moving average convergence divergence

# bollinger band
rstd20 = dataTrad['close'].rolling(window=20).std()
bollUpperband = mav20 + 2 * rstd20
bollLowerband = mav20 - 2 * rstd20

# CCI - Commodity Channel Index
techpr = (dataTrad['close'] + dataTrad['high'] + dataTrad['low']) / 3
cci = (techpr - techpr.rolling(14).mean())/(techpr.rolling(14).std())

# EMV - Ease of Movement 
dm = ((dataTrad['high'] + dataTrad['low']) / 2) - ((dataTrad['high'].shift(1) 
                                                    + dataTrad['low'].shift(1)) / 2)
br = (dataTrad['volume'] / 1000000) / ((dataTrad['high'] - dataTrad['low']))
evm = dm / br
evm_ma = evm.rolling(14).mean()

# ATR - Average True Range 
tr1 = (dataTrad['high'] - dataTrad['low'])
tr2 = abs(dataTrad['high'] - dataTrad['close'].shift(1))
tr3 = abs(dataTrad['low'] - dataTrad['close'].shift(1))
trRange = tr1
trRange[tr2>tr1] = tr2[tr2 > tr1]
trRange[tr3>trRange] = tr3[tr3 > trRange]
atr10 = trRange
atr10 = (atr10.shift(-1) * (10 - 1) + trRange)/10           # a corriger pour le lag

# ADX  -  Average Directionnal Index (relies on ATR)
tr1 = dataTrad['high'] - dataTrad['high'].shift(1)
tr2 = (dataTrad['low'].shift(1) - dataTrad['low'])
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

# RSI
diff = dataTrad['close'] - dataTrad['close'].shift(1)
upd = 0 * abs(diff)
downd = 0 * abs(diff)
upd[diff>0] = diff[diff>0]
downd[diff<0] = -diff[diff<0]
rs5 = upd.rolling(5).mean() / downd.rolling(5).mean()
rsi5 =  100 - (100 / (1 + rs5))
rs14 = upd.rolling(14).mean() / downd.rolling(14).mean()
rsi14 =  100 - (100 / (1 + rs14))

# === Adding parametrs to dataframe
# dataTrad['mav05'] = pd.Series(mav05, index = dataTrad.index)
# dataTrad['mav20'] = pd.Series(mav20, index = dataTrad.index)
dataTrad['mav05'] = pd.Series(mav05, index = dataTrad.index)
dataTrad['mav20'] = pd.Series(mav20, index = dataTrad.index)
dataTrad['expmav12'] = pd.Series(expmav12, index = dataTrad.index)
dataTrad['expmav26'] = pd.Series(expmav26, index = dataTrad.index)
dataTrad['expmav50'] = pd.Series(expmav50, index = dataTrad.index)
dataTrad['macd'] = pd.Series(macd, index = dataTrad.index)
dataTrad['BollUp20'] = pd.Series(bollUpperband, index = dataTrad.index)
dataTrad['BollDown20'] = pd.Series(bollLowerband, index = dataTrad.index)
dataTrad['std20'] = pd.Series(rstd20, index = dataTrad.index)
dataTrad['CCI14'] = pd.Series(cci, index = dataTrad.index)
dataTrad['EVM14'] = pd.Series(evm_ma, index = dataTrad.index)
dataTrad['ATR10'] = pd.Series(atr10, index = dataTrad.index)
dataTrad['+DX5'] = pd.Series(posdmi5, index = dataTrad.index)
dataTrad['-DX5'] = pd.Series(negdmi5, index = dataTrad.index)
dataTrad['ADX5'] = pd.Series(adx5, index = dataTrad.index)
dataTrad['ADX14'] = pd.Series(adx14, index = dataTrad.index)
dataTrad['RSI05'] = pd.Series(rsi5, index = dataTrad.index)
dataTrad['RSI14'] = pd.Series(rsi14, index = dataTrad.index)

# ====   Computing y  ============
print('Computing y')
dataTrad['yval'] = 0                                #reinitialize y value
#dataTrad['returnF'] = 0                             #reinitialize returnF (future return value)
for i in range(len(dataTrad)):
#for i in range (84466, 108627):
    if (i <= (len(dataTrad) - horiz - 1)                                                      # if row above the last number of rows for which indicators are not meaningfull
    and dataTrad.at[i,'Instrument'] == dataTrad.at[i+horiz, 'Instrument']):
        dataTrad.at[i, 'returnF'] = (dataTrad.at[i + horiz, 'close'] / dataTrad.at[i + 1, 'open']) - 1
    else:
        dataTrad.at[i, 'reindex'] = 0
        dataTrad.at[i, 'returnF'] = 0

    perfFut = dataTrad.at[i, 'returnF']
    if (perfFut < returnD):
       dataTrad.at[i, 'yval'] = -1
    elif (perfFut >= returnA) and (perfFut < returnB):
       dataTrad.at[i, 'yval'] = 1
    elif (perfFut >= returnB) and (perfFut < returnC):
       dataTrad.at[i, 'yval'] = 2
    elif (perfFut >= returnC):
       dataTrad.at[i, 'yval'] = 3
    else:
       dataTrad.at[i, 'yval'] = 0
    print(i , perfFut, dataTrad.at[i,'returnF'], dataTrad.at[i,'yval'])   

# ====   Saving enriched DB to SQL ============
print('Cleaning and saving data to SQL database')
#dataTrad.drop(dataTrad[(dataTrad["Instrument"] == ricIndex)].index, inplace=True)
dataTrad['reindex'] = dataTrad['reindex'].apply(lambda x: 0 if (x<mlp) else x)      #replace the index of first dat for later elimination with drop function
dataTrad.drop(dataTrad[dataTrad['reindex']==0].index, inplace=True)            #drop all rows where closing price is null
        
dataTrad.to_sql(dbSave, connect, if_exists='replace')   # replace the data if exists and saves the enriched DB to SQL
print('Enriched database saved')
    





