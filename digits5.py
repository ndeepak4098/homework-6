#
# read digits data
#


import numpy as np
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
import pandas as pd

print("+++ Start of pandas' datahandling +++\n")
# df here is a "dataframe":
df = pd.read_csv('digits5.csv', header=0) # read the file w/header row #0
df.head()                                 # first five lines
df.info()                                 # column details

def transform(s):
    """ 
    from number to string
    """
    return 'digit ' + str(s)
df['label'] = df['64'].map(transform)  # apply the function to the column

print("\n+++ End of pandas +++\n")



print("+++ Start of numpy/scikit-learn +++\n")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_data_orig = df.iloc[:,0:64].values # integer locations" of rows/cols
y_data_orig = df[ '64' ].values      # since there are 64 columns, get each of the columns
feature_names = df.columns.values    # get the names into a list! (not important)

X_data_full = X_data_orig[0:,]   # 2d array
y_data_full = y_data_orig[0:]    # 1d column

from matplotlib import pyplot as plt

indices = np.random.permutation(len(X_data_full))  # this scrambles the data each time
X_data_full = X_data_full[indices]
y_data_full = y_data_full[indices]

#
# The first ten will be our test set - the rest will be our training set
#
X_test = X_data_full[0:10,0:64]              # the final testing data out of 64
X_train = X_data_full[10:,0:64]              # the training data out of 64

y_test = y_data_full[0:10]                  # the 10 final testing outputs/labels (unknown)
y_train = y_data_full[10:]                  # the 10 training outputs/labels (known)


#max depth is where the loop provides the highest value of accuracy
max_depth = 13

#go through the loop 10 times from this range to see which max depth gives the most accuracy
#for max_depth in range(1,15):
dtree = tree.DecisionTreeClassifier(max_depth=max_depth)


for i in range(10):  #take the average cv testing score by cv_target_test/10
        
        #
        # split into our cross-validation sets...
        #
    cv_data_train, cv_data_test, cv_target_train, cv_target_test = \
    cross_validation.train_test_split(X_train, y_train, test_size=0.2) # random_state=0 

        # fit the model using the cross-validation data
        #   typically cross-validation is used to get a sense of how well it works
        #   and tune any parameters, such as max_depth here
    dtree = dtree.fit(cv_data_train, cv_target_train) 
    print("CV training-data score:", dtree.score(cv_data_train,cv_target_train))
    print("CV testing-data score:", dtree.score(cv_data_test,cv_target_test))


X_test = X_data_orig[0:9,0:64]              # the final testing data
X_train = X_data_orig[9:,0:64]              # the training data

y_test = y_data_orig[0:9]                  # the final testing outputs/labels (unknown) out of the 10
y_train = y_data_orig[9:]                  # the training outputs/labels (known) out of the 10

#
for max_depth in [1,2,3]:
    #
    # we'll use max_depth between 1 and 3
    #
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)

    # this next line is where the full training data is used for the model
    dtree = dtree.fit(X_data_full, y_data_full) 
    print("\nCreated and trained a decision tree classifier")  

    #
    # write out the dtree to tree.dot (or another filename of your choosing...)
    tree.export_graphviz(dtree, out_file='tree' + str(max_depth) + '.dot',  filled=True, rotate=False, leaves_parallel=True)  
    # the website to visualize the resulting graph (the tree) is at www.webgraphviz.com
    #


# here are some examples, printed out:
print("data_test's predicted outputs are")
print(dtree.predict(X_test))

# and here are the actual labels (iris types)
print("and the actual labels are")
print(y_test)


def show_digit( row ):
    """ input Pixels should be an np.array of 64 integers (from 0 to 15) 
        there's no return value, but this should show an image of that 
        digit in an 8x8 pixel square
    """
    Pixels = X_data_full[row:row+1,:]
    print("That image has the label:", y_data_full[row])
    Patch = Pixels.reshape((8,8))
    plt.figure(1, figsize=(4,4))
    plt.imshow(Patch, cmap=plt.cm.gray_r, interpolation='nearest')  # cm.gray_r   # cm.hot
    plt.show()