#*******************Imports***************************************************
from normalizer import Normalizer
from imports import *
#*******************Stop Imports**********************************************



#*******************Functions*************************************************
#________________________RMLSE Function____________________
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

#force brand_name, category_name, and item_conditon_id value types to be "catergory"
combined['brand_name'] = combined['brand_name'].astype('category') 
combined['category_name'] = combined['category_name'].astype('category')
combined['item_condition_id'] = combined['item_condition_id'].astype('category')

#force item_descpritom value type to be "string"
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

#________________________Preform KFold____________________\

y = np.log1p(train['price']) #y = natural log of train price variable (used here as the total number of elements)
kf = KFold(len(y), 10) # creates kfold that will "fold" the data 10 times (splits data into 10 parts)

#make iterator object out of Kfold (steps through each fold, sampling data), next returns each item
trainIndicies, validIndicies = next(iter(kf))

#trainSparse used in spliting of train and test data 
XTrain, yTrain = trainSparse[trainIndicies], y[trainIndicies]
XValid, yValid = trainSparse[validIndicies], y[validIndicies]

#*******************   Stop Main    ******************************************




