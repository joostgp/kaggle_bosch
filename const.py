# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:31:13 2016

@author: joostbloom
"""


BASE_PATH = '/Volumes/My Book/kaggle_bosch/'

DATA_PATH = '/Users/joostbloom/Documents/kaggle/bosch/data/'

LOG_PATH = '/Users/joostbloom/Documents/kaggle/bosch/logs/'

AUTHOR = 'joostgp'

TRAIN_FILES = ['train_numeric',
               'train_categorical_to_num',
               'train_date']
               
TEST_FILES = ['test_numeric',
              'test_categorical_to_num',
              'test_date']
              
LOOK_UP_TABLE = DATA_PATH + 'date_feat_lut_V2.csv'

CV = DATA_PATH + 'folds_V1.pkl'