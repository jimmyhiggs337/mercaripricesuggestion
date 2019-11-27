#*******************Imports***************************************************
from normalizer import Normalizer
from imports import *
#*******************Stop Imports**********************************************



#*******************Functions*************************************************
#________________________Ridge Recression Function____________________
#calculates error of MLA
def rmsle(y, y1):
    assert len(y) == len(y1)
    return i.np.sqrt(i.np.mean(i.np.power(i.np.log1p(y)-i.np.log1p(y1), 2)))

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
catVar = cv.fit_transform(combined['category_name'])
catVar

#apply count vectorizer to product name
cv = CountVectorizer(min_df=10) #ignores words that occur less than 10 times
name = cv.fit_transform(combined['name'])
name


#________________________TFIDF Vectorizer___________________
# TFIDF - More weight(importance) on "rare" words. Less weight(importance) on common words
tv = TfidfVectorizer(max_features=55000, ngram_range=(1, 2), stop_words='english')#builds a vocabylary of the top 55000 features (words), and limits the ngram to a unigram or a bigram 
itemDesc = tv.fit_transform(combined['item_description'])


#________________________Label Binarizer____________________
# Label Binarizer - Converts labels into numerical valus "a,b,c" -> [1,2,3]

#apply label binarizer to brand name
lb = LabelBinarizer(sparse_output=True) #returns array in sparse CSR format, allows fast row access
brand = lb.fit_transform(combined['brand_name'])

#________________________Create CSR Matrix____________________
# Create our final sparse matrix
dummyVar = csr_matrix(pd.get_dummies(combined[['item_condition_id', 'shipping']], sparse=True).values) #turnes values of item_condtion_id from a word to a value 1-3 and shippting from a word to a value of 1 or 0 aka "dummy values"

# Combine everything together
sparseMerge = hstack((dummyVar, itemDesc , brand, catVar, name)).tocsr() #creates CSR matrix (multidimensional array) of simplified dataset variables
trainSparse= sparseMerge[:trainSize] #creates a csr matrix for the train data seperate from the test data
testSparse= sparseMerge[trainSize:] #creates a csr matrix for the test data seperate from the train data
#________________________Preform KFold____________________
kf = KFold(len(y), round(1 / .10))
XTrain, yTrain = X_train_sparse[train_indicies], y[train_indicies]
X_valid, y_valid = X_train_sparse[valid_indicies], y[valid_indicies]

#*******************   Stop Main    ******************************************




