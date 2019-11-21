# -*- coding: utf-8 -*-
#*******************Imports**********************************************
import pandas as pd #library used to import and read data
import numpy as np #library used to import and read data
#import matplotlib.pyplot as plt
#import seaborn as sns
#import gc
import lightgbm as lgb #MLA import
# CountVectorizer - counts word frequencies 
# TFIDF - More weight(importance) on "rare" words. Less weight(importance) on common words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer # LabelBinarizer - Converts labels into numerical representation "a,b,c" -> [1,2,3]
from sklearn.linear_model import Ridge # Ridge - Reduces multicollinearity in regression. Applies L2 Regularization
from nltk.corpus import stopwords #list of stopwords for normalization
import string #python string class
#*******************Stop Imports**********************************************