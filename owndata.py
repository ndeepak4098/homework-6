#
# data from: https://vincentarelbundock.github.io/Rdatasets/csv/datasets/HairEyeColor.csv
# Hair = Hair Color, Eye = Eye Color, Freq= Frequency
# Hair: Black = 1, Brown = 2, Red = 3, Blond = 4
# Eye: Blue = 1, Brown = 2, Hazel = 3, Green = 4
#

import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
import pandas as pd

print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('owndata.csv', header=0)    # read the file w/header row #0
df.head()                                 # first five lines
df.info()                                 # column details

df = df.drop('Sex', axis=1)

print("\n+++ End of pandas +++\n")

print("+++ Start of numpy/scikit-learn +++\n")

print("Tests for Frequency")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
y_data_orig = df[ 'Freq' ].values      # individually addressable columns (by name)
X_data_orig = df.drop('Freq', axis=1).values 
feature_names = df.columns.values


X_data_full = X_data_orig[0:,:]  # make the 10 into 0 to keep all of the data
y_data_full = y_data_orig[0:]    # same for this line


#
# cross-validation and scoring to determine parameters...
# 

#
# we can scramble the data - but only if we know the test set's labels!
# 
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]

#
# The first ten will be our test set - the rest will be our training set
#
X_test = X_data_full             # the final testing data out of 64
X_train = X_data_full              # the training data out of 64

y_test = y_data_full                 # the 10 final testing outputs/labels (unknown)
y_train = y_data_full                 # the 10 training outputs/labels (known)

#
# cross-validation to determine the Decision Tree's parameter (to find max_depth)
#

from matplotlib import pyplot as plt

max_depth = 9
for x in [9]:

#for max_depth in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    dtree = tree.DecisionTreeClassifier(max_depth= max_depth)

    
    df_score_test = 0
    df_score_train = 0
    
    for i in range(10):
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0
        
        dtree = dtree.fit(cv_data_train, cv_target_train)
        
        df_score_train += dtree.score(cv_data_train,cv_target_train)
        df_score_test += dtree.score(cv_data_test,cv_target_test) 
    
    df_score_train = df_score_train/10
    df_score_test = df_score_test/10
    
    #print("CV training-data score:", df_score_train)
    print("DT CV testing-data score:", df_score_test)



#
for max_depth in [9]:
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

max_depth = 4
for x in [4]:
    
#or max_depth in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:

    
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
max_depth = 4
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
Results for frequency prediction:

DT CV testing-data score: 0.271428571429
Predicted outputs are:
[ 7 34  7  5  8  7  2  3  8 30  3  7 53  7  5 32 10 53  9 14 25 10  2 14  7
  5 34  5  9 30 25 32]

RF CV testing-data score: 0.228571428571
Predicted outputs are
[32  2  9  7  7 66 10 66  7 50 10 29 14 30  2  7  5  3 10 29 30  3  8 10  9
 10 14  8  5 10 32 50]

"""

print("Tests for Eye Color")

df = df.drop('Hair', axis=1)

y_data_orig = df[ 'Eye' ].values      # individually addressable columns (by name)
X_data_orig = df.drop('Eye', axis=1).values 
feature_names = df.columns.values


X_data_full = X_data_orig[0:,:]  # make the 10 into 0 to keep all of the data
y_data_full = y_data_orig[0:]    # same for this line


#
# cross-validation and scoring to determine parameters...
# 

#
# we can scramble the data - but only if we know the test set's labels!
# 
indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]

#
# The first ten will be our test set - the rest will be our training set
#
X_test = X_data_full             # the final testing data out of 64
X_train = X_data_full              # the training data out of 64

y_test = y_data_full                 # the 10 final testing outputs/labels (unknown)
y_train = y_data_full                 # the 10 training outputs/labels (known)

#
# cross-validation to determine the Decision Tree's parameter (to find max_depth)
#

from matplotlib import pyplot as plt

max_depth = 6
for x in [6]:

#for max_depth in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    dtree = tree.DecisionTreeClassifier(max_depth= max_depth)

    
    df_score_test = 0
    df_score_train = 0
    
    for i in range(10):
        cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
            cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0
        
        dtree = dtree.fit(cv_data_train, cv_target_train)
        
        df_score_train += dtree.score(cv_data_train,cv_target_train)
        df_score_test += dtree.score(cv_data_test,cv_target_test) 
    
    df_score_train = df_score_train/10
    df_score_test = df_score_test/10
    
    #print("CV training-data score:", df_score_train)
    print("DT CV testing-data score:", df_score_test)



#
for max_depth in [6]:
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
    
#for max_depth in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:

    
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
#print("feature importances:", rforest.feature_importances_)  


# here are some examples, printed out:
print("predicted outputs are")
print(rforest.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)

"""
Results for Eye Color prediction:

DT CV testing-data score: 0.385714285714
Predicted outputs are
[1 1 4 2 1 4 2 2 2 4 1 1 2 3 3 3 1 3 1 2 1 1 1 1 1 2 1 3 1 2 1 2]

RF CV testing-data score: 0.314285714286
Predicted outputs are
[1 4 4 2 4 4 2 2 2 4 1 1 2 3 3 3 1 3 3 2 3 1 3 3 3 2 1 3 1 1 1 2]

"""



"""
Overview:
My data set choice looks at the relationship between Hair Color, Eye Color, Sex and the frequency at which this occurs.
Using my data, I used data trees and random forests to test whether or not I could accurately predict (based on previous data)
someone's eye color, and the frequency at which traits (certain Hair, eye and sex) occur. What was interesting about my data was that
there wasn't as much of correlation between these items, which makes sense, because a lot of these factors are dependent on genetic things like 
race. However, there was some correlation, because hair color and eye color are more popular in some combinations. For this reason, the highest 
that my data could predict a lot of the time was around 30%.
"""