# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:31:13 2016

@author: joostbloom
"""
import pickle
import os
import const
import re
import datetime

import pandas as pd
import numpy as np


def load_data_file(f, ftype='pkl', nrows=None):
    
    if ftype=='csv':
        print('Returning {}.csv'.format(f))
        return pd.read_csv(os.path.join(const.BASE_PATH,f + '.csv'), nrows=nrows)
    elif ftype=='bin':
        with open(os.path.join(const.BASE_PATH,f + '_bin.pkl'),'rb') as f:
            print('Returning {}_bin.pkl'.format(f))
            return pickle.load(f)
    else:
        with open(os.path.join(const.BASE_PATH,f + '.pkl'),'rb') as f:
            print('Returning {}.pkl'.format(f))
            return pickle.load(f)

def read_last_column(csv_file):
    sample = pd.read_csv(os.path.join(const.BASE_PATH,csv_file), nrows=1)
    
    return pd.read_csv(os.path.join(const.BASE_PATH,csv_file), usecols=[0, sample.shape[1]-1], index_col=0)

def read_first_column(csv_file):
    
    return pd.read_csv(os.path.join(const.BASE_PATH, csv_file + '.csv'), usecols=[0])
    
def get_responses():
    y = read_last_column('train_numeric.csv')
    
    n_1 = y[y.Response==1].index.values
    n_0 = y[y.Response==0].index.values
    
    return y, n_1, n_0
       
def get_columns_csv(csv_file):
    ''' gets data columns for csv file identifier '''
    cols = list(pd.read_csv(os.path.join(const.BASE_PATH, csv_file + '.csv'), 
                              nrows=1,
                              index_col=0,
                             dtype=np.float32).columns)
    #print cols
    
    if 'Response' in cols:
        cols.remove('Response')
    
    
    return cols

def col_name_to_station_info(col_names):
    return [[int(i) for i in re.findall(r'\d+', col_name)] for col_name in col_names]

def get_station_info(csv_file):
    ''' extracts line and station info from list of columns '''
    
    cols = get_columns_csv(csv_file)
    
    station_info = [[int(i) for i in re.findall(r'\d+', col_name)] for col_name in cols]
    station_info = pd.DataFrame(station_info, columns=['line', 'station', 'feature_nr'])
    station_info['name'] = cols
    
    return station_info
    
def write_meta_info(meta_file, meta_info):
    ''' writes dict with meta info to text file'''
    
    meta_info['created_by'] = const.AUTHOR
    meta_info['created_at'] = str(datetime.datetime.now())
    
    with open(meta_file,'w') as f:

        for (key, value) in meta_info.iteritems():
            f.write('[{}]\n'.format(key))
            if isinstance(value, dict):
                for (k,v) in value.iteritems():
                    f.write('{}: {}\n'.format(k,v))
            else:
                f.write('{}\n'.format(value))
            f.write('\n')
    
    
