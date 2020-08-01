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
#↔ import numpy as np
# from sqlalchemy import create_engine

# ====   Connecting to Eikon Data API ============
cfg = cp.ConfigParser()
cfg.read('eikon.cfg')   # to be adjusted for file location
ek.set_app_key(cfg['eikon']['app_id'])

# ====   Connecting to SQL DB and loading lists ============
connect = sq.connect('StockList.db')        # connect to the DB FinancialDB
cursor = connect.cursor()
cursor.execute("SELECT field1 FROM TRlistCy")            # querry for a list of companies
outp = cursor.fetchall()
rics = list(chain(*outp))
cursor.execute("SELECT TRRIC FROM TRdataFinStat")               # querry for a list of fields to be populated by company
outp = cursor.fetchall()
listTR = list(chain(*outp))


sq_table = "TRdataEuroPriv"

# ====   Functions ============
for i in range(len(rics)):
# for i in range(11099, 11874):
    dataTR, err = ek.get_data(str(rics[i]), listTR, {'Scale': 3, 'SDate': 0, 'EDate': -40, 'FRQ': 'FY', 'Curn': 'EUR'})
    if len(dataTR) < 2:
        print('error', i)
    else:
        dataTR.drop(dataTR[pd.isnull(dataTR["Tot Assets"])].index, inplace=True)            #drop all rows where Total assets is null
        dataTR.to_sql(sq_table, connect, if_exists='append')   # add the data of company i to sql DB
        print(i, len(dataTR))
    





