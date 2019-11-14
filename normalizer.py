# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 02:27:22 2019

@author: ulgym
"""
# importing libraries
#import pandas as pd
from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize 
import string

set(stopwords.words('english'))
stopWords = stopwords.words('english')

class Normalizer:
    #def __init__(self):
        
    def removePunc(sentence: str) -> str:
        return sentence.translate(str.maketrans('', '', string.punctuation))
    
    def removeStopWords(words):
        wordsArr=words.split(' ')
        i=0
        while i < len(wordsArr): 
            if wordsArr[i].lower() in stopWords :
                wordsArr.pop(i)
            else:
                i=i+1            
        words= ' '.join(wordsArr)
        return words
    
    def toLower(words):
        return words.lower()
    
    def missingValues(df,col,missVal):
        df[col].fillna(value=missVal, inplace=True)
        
    
    
    
    """
    #testing
    testSet = pd.read_csv('train short.tsv', delimiter='\t')
    itemDesc= testSet["item_description"].to_string()
    itemDescNoStop = removeStopWords(itemDesc)
    itemDescNoPunc = removePunc(itemDesc)
    print(itemDesc.split('\n')[1])
    print(itemDescNoStop.split('\n')[1])
    print(itemDesc.split('\n')[74])
    print(itemDescNoPunc.split('\n')[74])
    print(testSet['brand_name'])
    missingValues(testSet,'brand_name', 'None')
    print(testSet['brand_name'])
    """