# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:26:35 2019

@author: Peisong Yang
"""
# 1. SQL Query to load Static Bond ----------------------------------------
def SQLQueryStatic():
    return ("SELECT * FROM bond_static")

# 2. Function to load bond prices from SQL --------------------------------
def SQLReturnBondPrice(ISIN):
    # constructing SQL Query
    ISINQuery = "SELECT * FROM bond_prices WHERE isin_id = '"+str(ISIN)+"'"
    return (ISINQuery)
    

# 3. SQL create a new table: bond_returns ----------------------------------
def SQLCreateRetTable():
    Command = "CREATE TABLE bond_returns (\
    cob_date TEXT NOT NULL,\
    clean_ret INTEGER,\
    dirty_ret INTEGER,\
    dirty_sd INTEGER,\
    dirty_avg INTEGER,\
    isin_ret TEXT,\
    FOREIGN KEY (isin_ret) REFERENCES bond_static(isin))"
    return (Command)

# 4. SQL Delete a table ---------------------------------------------------
def SQLDeleteTable():
    return("DROP TABLE bond_returns")