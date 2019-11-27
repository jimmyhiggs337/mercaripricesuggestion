# -*- coding: utf-8 -*-
#*******************Imports**********************************************
import pandas as pd #library used to import and read data
import numpy as np #library used to import and read data
#import matplotlib.pyplot as plt
#import seaborn as sns
#import gc
# vstack - adds rows, hstack - adds columns
# csr_matrix - used to handle sparse matrix
from scipy.sparse import vstack, hstack, csr_matrix
import lightgbm as lgb #MLA import
# CountVectorizer - counts word frequencies 
# TFIDF - More weight(importance) on "rare" words. Less weight(importance) on common words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer # LabelBinarizer - Converts labels into numerical representation "a,b,c" -> [1,2,3]
from sklearn.linear_model import Ridge # Ridge - Reduces multicollinearity in regression. Applies L2 Regularization
from nltk.corpus import stopwords #list of stopwords for normalization
from sklearn.cross_validation import KFold #randomly samples data to check accuracy of data
import string #python string class
#*******************Stop Imports**********************************************