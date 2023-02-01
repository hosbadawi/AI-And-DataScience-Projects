'''import the needed modules here:'''
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
pass

'''
In order to have reproducible results, you need to set the 
random states (a.k.a seeds) of the sources of randomness to
a constant number. You can set the local seeds to a constant
number in the respective place. If you have global seeds, 
set them here:
'''
pass

def train_then_test(x_train, y_train, x_test):
    '''
    This functions has the following inputs:
     * x_train: which is a numpy list of the training data
     * y_train: which is a numpy list of the corresponding labels for the given features
     * x_test: which is a numpy list test data


    Within the body of this function you need to build and train a
    classifier for the given training data (x_train, y_train) and
    then return the predicted labels for x_test.
    Output: predicted_labels (can be a python list, numpy list, or pandas list of the predicted labels)

    Notes:
        Do not change the name of this function.
        Do not add new parameters to this function.
        Do not remove the parameters of this function (x_train, y_train, x_test)
        Do not change the order of the parameters of this function
        If x_test contains 100 rows of data, predicted_labels needs to have 100 rows of predicted labels
    '''
    
    warnings.filterwarnings('ignore')
    
    ################################################################## first model
    clf1 = ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.35, min_samples_leaf=3, min_samples_split=16, n_estimators=100, random_state = 42)
    clf2 = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.55, min_samples_leaf=4, min_samples_split=18, n_estimators=100, random_state = 42)

    clf1.fit(x_train, y_train)

    x_train_temp1 = x_train.copy()
    x_train_temp1 = pd.concat([pd.DataFrame(x_train_temp1), pd.DataFrame(clf1.predict(x_train)), pd.DataFrame(clf1.predict_proba(x_train))], axis=1, ignore_index=True).astype(float)

    x_test_temp1 = x_test.copy()
    x_test_temp1 = pd.concat([pd.DataFrame(x_test_temp1), pd.DataFrame(clf1.predict(x_test)), pd.DataFrame(clf1.predict_proba(x_test))], axis=1, ignore_index=True).astype(float)

    clf2.fit(x_train_temp1, y_train)
    y_pred1 = clf2.predict(x_test_temp1)
    y_pred_proba1 = clf2.predict_proba(x_test_temp1)

    ################################################################## second model
    clf3 = MLPClassifier(alpha=0.0001, learning_rate_init=0.1, random_state = 42)
    clf4 = MLPClassifier(alpha=0.0001, learning_rate_init=0.1, random_state = 42)

    ICA_obj1 = FastICA(tol=0.4, random_state = 42)
    x_trainICA1 = ICA_obj1.fit_transform(x_train)
    x_testICA1 = ICA_obj1.transform(x_test)

    clf3.fit(x_trainICA1, y_train)
    x_train_temp1 = x_train.copy()
    x_train_temp1 = pd.concat([pd.DataFrame(x_train_temp1), pd.DataFrame(clf3.predict(x_trainICA1)), pd.DataFrame(clf3.predict_proba(x_trainICA1))], axis=1, ignore_index=True).astype(float)

    x_test_temp1 = x_test.copy()
    x_test_temp1 = pd.concat([pd.DataFrame(x_test_temp1), pd.DataFrame(clf3.predict(x_testICA1)), pd.DataFrame(clf3.predict_proba(x_testICA1))], axis=1, ignore_index=True).astype(float)

    clf4.fit(x_train_temp1, y_train)
    y_pred2 = clf4.predict(x_test_temp1)
    y_pred_proba2 = clf4.predict_proba(x_test_temp1)

    ################################################################## third model
    clf5 = ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.95, min_samples_leaf=5, min_samples_split=7, n_estimators=100, random_state = 42)
    clf6 = DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=18, min_samples_split=19, random_state = 42)

    ICA_obj1 = FastICA(tol=0.25, random_state = 42)
    x_trainICA1 = ICA_obj1.fit_transform(x_train)
    x_testICA1 = ICA_obj1.transform(x_test)

    clf5.fit(x_trainICA1, y_train)
    x_train_temp1 = x_train.copy()
    x_train_temp1 = pd.concat([pd.DataFrame(x_train_temp1), pd.DataFrame(clf5.predict(x_trainICA1)), pd.DataFrame(clf5.predict_proba(x_trainICA1))], axis=1, ignore_index=True).astype(float)

    x_test_temp1 = x_test.copy()
    x_test_temp1 = pd.concat([pd.DataFrame(x_test_temp1), pd.DataFrame(clf5.predict(x_testICA1)), pd.DataFrame(clf5.predict_proba(x_testICA1))], axis=1, ignore_index=True).astype(float)

    clf6.fit(x_train_temp1, y_train)
    y_pred3 = clf6.predict(x_test_temp1)
    y_pred_proba3 = clf6.predict_proba(x_test_temp1)

    ########################################################################### TEST
    class0 = pd.DataFrame(y_pred_proba2).iloc[:,0]
    class1 = pd.DataFrame(y_pred_proba1).iloc[:,1]

    y_pred_final = pd.concat([pd.DataFrame(class0), pd.DataFrame(class1), pd.DataFrame(y_pred_proba3)], axis=1, ignore_index=True).astype(float)

    temp = []
    for i in range(len(y_pred_final)):
        if (y_pred_final.iloc[i, 0] + y_pred_final.iloc[i, 2]) > (y_pred_final.iloc[i, 1] + y_pred_final.iloc[i, 3]):
            temp.append(0)
        elif (y_pred_final.iloc[i, 0] + y_pred_final.iloc[i, 2]) < (y_pred_final.iloc[i, 1] + y_pred_final.iloc[i, 3]):
            temp.append(1)
        else:
            print("error")                          
        
    temp = pd.DataFrame(temp)
    print(temp.value_counts())

    temp = temp.to_numpy()

    return temp
    pass

if __name__ == '__main__':
    # You need to do all the function calls for testing purposes in this scope:
        
    # read the data
    x_train = pd.read_csv("x_train.csv")
    y_train = pd.read_csv("y_train.csv")
    x_test = pd.read_csv("x_test.csv")
    y_train = y_train.values.ravel()
    x_test = x_test.to_numpy()
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    
    # shuffle the data
    np.random.seed(42)
    idx = np.random.permutation(x_train.index)
    x_train = x_train.reindex(idx)
    y_train = y_train.reindex(idx)
    x_train = x_train.reset_index(drop = True).to_numpy()
    y_train = y_train.reset_index(drop = True).to_numpy()
    del idx
    
    # call the function
    y_pred = train_then_test(x_train, y_train, x_test)

    # # generate the csv file
    # ids = np.arange(len(y_pred))
    # ids = pd.DataFrame(ids, columns=['Id'])
    # y_pred = pd.DataFrame(y_pred, columns=['Predicted'])
    # submition = pd.concat([ids, y_pred], axis=1, ignore_index=True).astype(int)
    # submition.columns = ["Id", "Predicted"]
    # submition.to_csv('300327269.csv', index=False)
    pass