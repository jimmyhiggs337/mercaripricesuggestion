# -*- coding: utf-8 -*-
#*******************Imports**********************************************
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
import Normalizer
#*******************Stop Imports**********************************************





#*******************Functions*************************************************

#________________________Ridge Recression Function____________________
#calculates error of MLA
def rmsle(y, y1):
    assert len(y) == len(y1)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y1), 2)))

#***********************Stop Functions****************************************
    





#*******************   Main    ************************************************

#________________________Import Data________________________
train = pd.read_csv('train.tsv', sep = '\t')
train.head()
test = pd.read_csv('test.tsv', sep = '\t',engine = 'python')
combined = pd.concat([train,test])
submission = test[['test_id']]
trainSize = len(train)

#________________________Data Normailization________________________

#removing missing values
Normalizer.missingValues(combined,'brand_name', 'None')
Normalizer.missingValues(combined,'item_description', 'None')
Normalizer.missingValues(combined,'category_name', 'missing')

combined.item_description = combined.item_description.astype(str)

#removing punctuation from item description
combined.item_description = combined['item_description'].apply(Normalizer.removePunc)

#removing stop words from item description
combined.item_description = combined['item_description'].apply(Normalizer.removeStopWords)

#setting words to lowercase
combined.item_description = combined['item_description'].apply(Normalizer.toLower)

#combined_ML = combined.sample(frac=0.1).reset_index(drop=True)




#________________________Count Vectorizer___________________
# Count Vectorizer - counts word frequencies 

#apply count vectorizer to category name
cv = CountVectorizer()
catName = cv.fit_transform(combined['category_name'])
catName

#apply count vectorizer to product name
cv = CountVectorizer(min_df=10)
name = cv.fit_transform(combined['name'])
name


#________________________TFIDF Vectorizer___________________
# TFIDF - More importance/weights on "rare" words. Less importance/weights on "frequent" words
tv = TfidfVectorizer(max_features=55000, ngram_range=(1, 2), stop_words='english')
itemDesc = tv.fit_transform(combined['item_description'])



#________________________Label Binarizer____________________
# Label Binarizer - Converts labels into numerical representation "a,b,c" -> [1,2,3]
lb = LabelBinarizer(sparse_output=True)
brand = lb.fit_transform(combined['brand_name'])

#*******************   Stop Main    ******************************************




