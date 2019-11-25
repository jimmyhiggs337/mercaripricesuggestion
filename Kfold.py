# -*- coding: utf-8 -*-
#use KFold for cross validataion 
from sklearn.cross_validation import KFold
eval_size = .10
#y is the log price variable used here as the total number of elements
#10 folds (1/.1=10) 
kf = KFold(len(y), round(1 / eval_size))
#make iterator object out of Kfold, next to return each item
train_indicies, valid_indicies = next(iter(kf))
#X_train_sparse used in spliting of train and test data 
X_train, y_train = X_train_sparse[train_indicies], y[train_indicies]
X_valid, y_valid = X_train_sparse[valid_indicies], y[valid_indicies]
