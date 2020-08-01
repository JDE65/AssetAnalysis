# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:59:03 2020

@author: JDE65
"""
# ====   Installing libraries ============
import eikon as ek
import cufflinks as cf
import configparser as cp
cf.set_config_file(offline=True)
import pandas as pd
import sqlite3 as sq
from itertools import chain

# ====   Connecting to Eikon Data API ============
cfg = cp.ConfigParser()
cfg.read('eikon.cfg')   # to be adjusted for file location
ek.set_app_key(cfg['eikon']['app_id'])

# ====   Connecting to SQL DB and loading lists ============
connect = sq.connect('CorrelatedAssets.db')        # connect to the DB FinancialDB
cursor = connect.cursor()
startDate = -8000
endDate = -1
# Fill-in Indexes
cursor.execute("SELECT RIC FROM CorrAssIndex")            # querry for a list of companies
outp = cursor.fetchall()
rics = list(chain(*outp))
sq_table = "dataCorrIndex"
for i in range(len(rics)):
    dataTR, err = ek.get_data(str(rics[i]),["TR.PriceCloseDate", "TR.PriceClose", "TR.PriceOpen", "TR.PriceHigh", "TR.PriceLow"],
                              {'SDate': startDate, 'EDate' : endDate, 'FRQ': 'D'})
    if len(dataTR) < 2:
        print('error', i)
    else:
        dataTR.drop(dataTR[pd.isnull(dataTR["Price Close"])].index, inplace=True)            #drop all rows where closing price is null
        
        dataTR.to_sql(sq_table, connect, if_exists='append')   # add the data of company i to sql DB
        print(i, len(dataTR))

# Fill-in EURIBOR & LIBOR
cursor.execute("SELECT RIC FROM CorrAssIBOR")            # querry for a list of companies
outp = cursor.fetchall()
rics = list(chain(*outp))
sq_table = "dataCorrIBOR"
for i in range(len(rics)):
    dataTR, err = ek.get_data(str(rics[i]),["TR.FIXINGVALUE.date", "TR.FIXINGVALUE"], 
                          {'SDate': startDate, 'EDate' : endDate, 'FRQ': 'D'})
    if len(dataTR) < 2:
        print('error', i)
    else:
        dataTR.drop(dataTR[pd.isnull(dataTR["Fixing Value"])].index, inplace=True)            #drop all rows where closing price is null
        
        dataTR.to_sql(sq_table, connect, if_exists='append')   # add the data of company i to sql DB
        print(i, len(dataTR))

# Fill-in Others
cursor.execute("SELECT RIC FROM CorrAssFXO")            # querry for a list of companies
outp = cursor.fetchall()
rics = list(chain(*outp))
sq_table = "dataCorrFXO"
for i in range(len(rics)):
    dataTR, err = ek.get_data(str(rics[i]),["TR.MIDPRICE.date", "TR.MIDPRICE"],
                              {'SDate': startDate, 'EDate' : endDate, 'FRQ': 'D'})
    if len(dataTR) < 2:
        print('error', i)
    else:
        dataTR.drop(dataTR[pd.isnull(dataTR["Mid Price"])].index, inplace=True)            #drop all rows where closing price is null
        
        dataTR.to_sql(sq_table, connect, if_exists='append')   # add the data of company i to sql DB
        print(i, len(dataTR))

print("Database filled in")
    





