#
# read Titanic data
#


import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
from sklearn import datasets
import pandas as pd

print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('titanic5.csv', header=0)    # read the file w/header row #0
df.head()                            # first five lines
df.info()                                 # column details

#
# drop columns here
#
df = df.drop('name', axis=1)  # everything except the 'survival' column
df = df.drop('ticket', axis=1)
df = df.drop('embarked', axis=1)
df = df.drop('home.dest', axis=1)
df = df.drop('age', axis=1)
df = df.drop('sibsp', axis=1)
df = df.drop('parch', axis=1)
df = df.drop('cabin', axis=1)

df = df.dropna() #drop empty rows



# One important one is the conversion from string to numeric datatypes!
# You need to define a function, to help out...
def tr_mf(s):
    """ from string to number
    """
    d = { 'male':0, 'female':1 }
    return d[s]

df['sex'] = df['sex'].map(tr_mf)  # apply the function to the column
#
# end of conversion to numeric data...


print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")

#
# Decision Tree!
# 

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays =     
y_data_full = df[ 'survived' ].values  
X_data_full = df.drop('survived', axis=1).values     
feature_names = df.columns.values

X_data_full = X_data_full[0:,:]  # 2d array
y_data_full = y_data_full[0:]    # 1d coloumn

# indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
# X_data_full = X_data_full[indices]
# y_data_full = y_data_full[indices]

#
# The first ten will be our test set - the rest will be our training set
#
X_test = X_data_full[0:42,:]              # the final testing data out of 64
X_train = X_data_full[42:,:]              # the training data out of 64

y_test = y_data_full[0:42]                  # the 10 final testing outputs/labels (unknown)
y_train = y_data_full[42:]                  # the 10 training outputs/labels (known)


#max depth is where the loop provides the highest value of accuracy
#go through the loop 10 times from this range to see which max depth gives the most accuracy

from matplotlib import pyplot as plt

max_depth = 3

    
dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
for x in [3]:
    
    df_score_test = 0
    df_score_train = 0
    
    for i in range(10):
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 


        # fit the model using the cross-validation datasets
        #   typically cross-validation is used to get a sense of how well it works
        #   and tune any parameters, such as max_depth here
        dtree = dtree.fit(cv_data_train, cv_target_train)
        
        df_score_train += dtree.score(cv_data_train,cv_target_train)
        df_score_test += dtree.score(cv_data_test,cv_target_test) 
    
    df_score_train = df_score_train/10
    df_score_test = df_score_test/10
    
    #print("CV training-data score:", df_score_train)
    print("DT CV testing-data score:", df_score_test)

# y_test = y_data_full[0:9]                  # the final testing outputs/labels (unknown) out of the 10
# y_train = y_data_full[9:]                  # the training outputs/labels (known) out of the 10



#
for max_depth in [3]:
    #
    # we'll use max_depth between 1 and 3
    #
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)

    # this next line is where the full training data is used for the model
    dtree = dtree.fit(X_data_full, y_data_full) 
    print("\nCreated and trained a decision tree classifier")  

    #
    # write out the dtree to tree.dot (or another filename of your choosing...)
    
    tree.export_graphviz(dtree, out_file='tree' + str(max_depth) + '.dot',   # constructed filename!
                            feature_names=feature_names,  filled=True, rotate=False, # LR vs UD
                            leaves_parallel=True)  
    
    tree.export_graphviz(dtree, out_file='tree' + str(max_depth) + '.dot',  filled=True, rotate=False, leaves_parallel=True)  
    # the website to visualize the resulting graph (the tree) is at www.webgraphviz.com
    #

# here are some examples, printed out:
print("predicted outputs are")
print(dtree.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)


#
# Random Forests!
# 

max_depth = 7
for x in [7]:
    
    #for max_depth in range(1,15):

    
    rforest = ensemble.RandomForestClassifier(max_depth=max_depth, n_estimators=10)
    # adapt for cross-validation (at least 10 runs w/ average test-score)
    
    rf_score_test = 0
    rf_score_train = 0
    
    for i in range(10):
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 
        
        rforest = rforest.fit(cv_data_train, cv_target_train) 
        
        rf_score_train += rforest.score(cv_data_train,cv_target_train)
        rf_score_test += rforest.score(cv_data_test,cv_target_test) 
    
    rf_score_train = rf_score_train/10
    rf_score_test = rf_score_test/10
    
    #print("CV training-data score:", rf_score_train)
    print("RF CV testing-data score:", rf_score_test)

# rforest.estimators_  [a list of dtrees!]


#
# we'll use max_depth == biggest value from above
#
max_depth = 7
rforest = ensemble.RandomForestClassifier(max_depth=max_depth, n_estimators=100)

#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#

# this next line is where the full training data is used for the model
rforest = rforest.fit(X_train, y_train) 
print("\nCreated and trained a randomforest classifier") 

#
# feature importances
#
print("feature importances:", rforest.feature_importances_)  


# here are some examples, printed out:
print("predicted outputs are")
print(rforest.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)


"""
Results:
DT CV testing-data score: 0.81062992126
DT predicted results for survival:
[0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 1 0 0 0
 1 1 0 0 1]
This prediction got 7 wrong
The first 2 lines of the DT ask whether the line of the data read the csv file, and drop
unimportant columns that are not related to the question (either survial or age). Then, the df model
basically takes in the data we are looking at in the y data, and excludes it from the x data.

RF CV testing-data score: 0.811023622047
RF predicted results for survival:
[0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 1 0 1 0 1 0 1 0 1 0 1 0 0 1 1 1 0 0 0
 1 1 0 0 1]
 This prediction got 6 wrong
"""

#The impute part of this problem is in the other file labled titanic5_1.py