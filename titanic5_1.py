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
df.info()                            # column details

#
# drop columns here21
#
df = df.drop('name', axis=1)  # everything except the 'survival' column
df = df.drop('ticket', axis=1)
df = df.drop('embarked', axis=1)
df = df.drop('home.dest', axis=1)
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
y_data_full = df[ 'age' ].values  
X_data_full = df.drop('age', axis=1).values     
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


#
# Random Forests!
# 

max_depth = 4
for x in [4]:
    #for max_depth in range(1,15):
    
    rforest = ensemble.RandomForestRegressor(max_depth=max_depth, n_estimators=10)
    
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
rforest = ensemble.RandomForestRegressor(max_depth=max_depth, n_estimators=100)

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
print("iris_X_test's predicted outputs are")
print(rforest.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)


"""
Results:
The RF testing-data score is 0.202912560273-
iris_X_test's predicted outputs are:
[ 27.90119133  27.90119133  27.90119133  26.94039815  27.93970478
  29.40956365  19.1440891   27.90119133  27.93970478  27.90119133
  33.69552132  33.40914523  25.46040562  37.7708308   26.16715548
  26.16715548  28.28932669  34.37717433  24.94288448  23.61776456
  35.68219422  35.43657207  42.99561846  42.37316587  42.99561846
  40.26067362  41.47362875  43.73781288  36.95848503  41.51619038
  51.51319293  35.93423069  36.34768203  42.13888309  37.7162605
  48.68419618  35.08672316  41.51988187  43.75913294  36.10719543
  36.13198876  37.7162605 ]

The RF model for age did not do nearly as well as the RF model for survival rates. I am able
to measure this based on the testing data percentage, and seeing how many each data got wrong.
With the RF model for survival, the data only got 6 wrong, vs with age, the data had much more wrong.
In addition, the highest data I could for age get was 20% vs with survival rate was 81%. Therefore,
the RF data for surival was more accurate.
"""