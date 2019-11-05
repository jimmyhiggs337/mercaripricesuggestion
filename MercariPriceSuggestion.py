# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
import gc
import lightgbm as lgb
# CountVectorizer - counts word frequencies 
# TFIDF - More importance/weights on "rare" words. Less importance/weights on "frequent" words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# LabelBinarizer - Converts labels into numerical representation "a,b,c" -> [1,2,3]
from sklearn.preprocessing import LabelBinarizer
# Ridge - Reduces multicollinearity in regression. Applies L2 Regularization
from sklearn.linear_model import Ridge

#________________________Import Data________________________
train = pd.read_csv('train.tsv', sep = '\t')
train.head()
test = pd.read_csv('test.tsv', sep = '\t',engine = 'python')
combined = pd.concat([train,test])
submission = test[['test_id']]
train_size = len(train)
#combined_ML = combined.sample(frac=0.1).reset_index(drop=True)



#________________________Count Vectorizer___________________
# Count Vectorizer - counts word frequencies 
cv = CountVectorizer()
X_category = cv.fit_transform(combined['category_name'])
#X_sub1 = cv.fit_transform(combined['sub_category_1'])
#X_sub2 = cv.fit_transform(combined['sub_category_2'])
#X_category
cv = CountVectorizer(min_df=10)
X_name = cv.fit_transform(combined['name'])



#________________________TFIDF Vectorizer___________________
# TFIDF - More importance/weights on "rare" words. Less importance/weights on "frequent" words
tv = TfidfVectorizer(max_features=55000, ngram_range=(1, 2), stop_words='english')
X_description = tv.fit_transform(combined['item_description'])



#________________________Label Binarizer____________________
# Label Binarizer - Converts labels into numerical representation "a,b,c" -> [1,2,3]
lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(combined['brand_name'])

#________________________Ridge Recression Function____________________
#calculates error of MLA
def rmsle(y, y1):
    assert len(y) == len(y1)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y1), 2)))