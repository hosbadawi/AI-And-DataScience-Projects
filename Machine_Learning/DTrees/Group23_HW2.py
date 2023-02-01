from sklearn.datasets import load_wine
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns , re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import time

class Assignment2:
    
    # Function to read the Dataset
    @staticmethod
    def readDataSet(DataSet_name, Sheet_Name, ColNames):
        Extension = re.findall('((.csv)|(.xls)|(.xlsx))', DataSet_name)
        Extension = str(Extension)
        if '.csv' in Extension:
            Extension = '.csv'
        elif '.xls' in Extension:
            Extension = '.xls'
        elif '.xlsx' in Extension:
            Extension = '.xlsx'
            
        if Extension == ('.xls' or '.xlsx'):
            Assignment2.DataFrame = pd.read_excel(DataSet_name, sheet_name = Sheet_Name)
        elif Extension == '.csv':
            Assignment2.DataFrame = pd.read_csv(DataSet_name , names = ColNames)
        return Assignment2.DataFrame
    
    # to generate Confusion Matrix
    @staticmethod
    def ConfusionMatrix(Y_Actual, Y_Pred):
        CF = confusion_matrix(Y_Actual, Y_Pred)
        return CF
    
    # to Plot Confusion Matrix
    @staticmethod
    def PLOT_ConfusionMatrix(CF,Title):
        sns.heatmap(CF, annot=True, fmt='d')
        plt.title(Title, fontsize = 15)
        plt.xlabel('Predicted', fontsize = 15)
        plt.ylabel('Actual', fontsize = 15)
        return plt.show()
    
    # Accuracy on Train
    @staticmethod
    def AccuracyTrain(Classifier,X_Train,Y_Train):
        return Classifier.score(X_Train, Y_Train) * 100
    
    # Accuracy on Test
    @staticmethod
    def AccuracyTest(Y_Actual, Y_Pred):
        return accuracy_score(Y_Actual, Y_Pred) * 100
    
    # Plot Decision Boundaries
    @staticmethod
    def Boundaries(X, Classifier, Title , numberOfClasses):
        colorsOptions = ['#FF00AE','blue','#8941FF','#00FF0F','#FF0000','#000000','#0B9397','#1B49CD','#11AF29','#560078','#ED036A']
        colors = colorsOptions[0:numberOfClasses]
        
        plt.figure(figsize=(8, 8), dpi=80)
        
        X_Axis = X.values
        X1, Y1 = np.meshgrid(np.arange(start = X_Axis[: ,0].min() - 0.1, stop = X_Axis.max() + 0.1, step = 0.01),
                             np.arange(start = X_Axis[: ,1].min() - 0.1, stop = X_Axis.max() + 0.1, step = 0.01))
        
        plt.contourf(X1, Y1, Classifier.predict(np.array([X1.ravel(), Y1.ravel()]).T).reshape(X1.shape),
                     alpha = 0.2, cmap =  ListedColormap(colors))
        
        plt.xlim(X1.min(), X1.max())
        plt.ylim(Y1.min(), Y1.max())
        plt.title(Title, fontsize = 15)
        return plt
    # Plot the Different Classes
    
    @staticmethod
    def PlotData(numberOfClasses, ListOFClasses):
        colorsOptions = ['#FF00AE','blue','#8941FF','#00FF0F','#FF0000','#000000','#0B9397','#1B49CD','#11AF29','#560078','#ED036A']
        MarkersOptions = ['*','^','v','P','D','x','X','h','H','d']
        colors = colorsOptions[0:numberOfClasses]
        Markers = MarkersOptions[0:numberOfClasses]
        
        for i in range(0,numberOfClasses):
            for j in range(0, len(ListOFClasses[i])):
                plt.annotate('0' if i==0 else '1' if i==1 else '2' if i==2 else '' , (ListOfClasses2Fea[i].values.tolist()[j][0] , ListOfClasses2Fea[i].values.tolist()[j][1]))
            plt.scatter(x = ListOFClasses[i].iloc[:, 0:1], y = ListOFClasses[i].iloc[:, 1:2], c=colors[i], marker = Markers[i], s=50)
        plt.xlabel('hue', fontsize = 15)
        plt.ylabel('Proline', fontsize = 15)
        return plt.show()
    
    # Slice the DataFrame based on The Different Classes (List Of DataFrames)
    @staticmethod
    def GetListOfClasses(numberOfClasses, DataSet, TargetColumn):
        ls = [None] * numberOfClasses
        for i in range(0,numberOfClasses):
            ls[i] = DataSet.loc[DataSet[TargetColumn] == i]
        return ls
    
    # Function to Plot PairPlot
    @staticmethod
    def PairPLot(DataSet , TargetVariable):
        return plt.show(sns.pairplot(DataSet, hue = TargetVariable, palette=sns.color_palette("husl", 3)))
    
    # Function to detemine which features will be eliminated
    @staticmethod
    def FeatureSelection(X,Y):
        Selector = ExtraTreesClassifier(n_estimators=2)
        Selector = Selector.fit(X, Y)
        return Selector.feature_importances_
    
    # Function to Encode Labels of Categorical data
    @staticmethod
    def labelEncoder(DataFrame , Categorical_Column, ListOfClassesNames):
        for i in range(len(ListOfClassesNames)):
            DataFrame.loc[DataFrame[str(Categorical_Column)] == ListOfClassesNames[i], str(Categorical_Column)] = i
        return DataFrame

#------------------------------------------------------------------------------------MAIN------------------------------------------------------------------------------------
# 1(a) -------------------------------------- Read DataSet
DataSet = load_wine()
X = DataSet.data
Y = DataSet.target

# Feature Scaling
Scaler = StandardScaler()
X = Scaler.fit_transform(X)
X = pd.DataFrame(X)
X.columns = DataSet.feature_names
DataSet = pd.concat([pd.DataFrame(X), pd.DataFrame(Y)],axis=1 , ignore_index = True)

# Create Object form our class
Assignment2 = Assignment2()

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

# GaussianNB Model
Classifier0 = GaussianNB()
Classifier0.fit(X_train, y_train)
y_pred0 = Classifier0.predict(X_test)

# Accuracy on Train SVM and PERCEP
Acc_OnTrain0 = Assignment2.AccuracyTrain(Classifier0, X_train, y_train)

# Accuracy on Test SVM and PERCEP
Acc_OnTest0 = Assignment2.AccuracyTest(y_test, y_pred0)

# Confusion matrix
CF_Matrix0 = Assignment2.ConfusionMatrix(y_test, y_pred0)
Assignment2.PLOT_ConfusionMatrix(CF_Matrix0,'GaussianNB')

# 1(b) -------------------------------------- classification_report
target_names = ['0', '1', '2']
print(classification_report(y_test, y_pred0, target_names = target_names))

# 1(c) -------------------------------------- Select two features for plotting using pair plot and 
print(Assignment2.FeatureSelection(X_train, y_train))
# Assignment2.PairPLot(DataSet, DataSet.columns[13])

# Plot Decision boundaries
X_train_2fea = X_train.iloc[:,[10,12]] # based on pair plot
X_test_2fea = X_test.iloc[:,[10,12]]
Test2Fea = pd.concat([pd.DataFrame(X_test_2fea), pd.DataFrame(y_test)],axis=1 , ignore_index = True)
Classifier1 = GaussianNB()
Classifier1.fit(X_train_2fea, y_train)

# GaussianNB Model
y_pred1 = Classifier1.predict(X_test_2fea)
# Accuracy on Train SVM and PERCEP
Acc_OnTrain1 = Assignment2.AccuracyTrain(Classifier1, X_train_2fea, y_train)
# Accuracy on Test SVM and PERCEP
Acc_OnTest1 = Assignment2.AccuracyTest(y_test, y_pred1)
# Confusion matrix
CF_Matrix1 = Assignment2.ConfusionMatrix(y_test, y_pred1)
Assignment2.PLOT_ConfusionMatrix(CF_Matrix1,'GaussianNB 2 features')
# classification_report
target_names = ['0', '1', '2']
print(classification_report(y_test, y_pred1, target_names = target_names))

ListOfClasses2Fea = Assignment2.GetListOfClasses(3, Test2Fea, pd.DataFrame(Test2Fea).columns[2])
Assignment2.Boundaries(X_test_2fea,Classifier1,'GaussianNB 2 features',3)
Assignment2.PlotData(3, ListOfClasses2Fea)

#------------------------------------------------------------------------------------KNN------------------------------------------------------------------------------------
# 2(a) -------------------------------------- 
CarDataSet = Assignment2.readDataSet('car_evaluation.csv', 'car_evaluation' , ColNames = ['Buying','Maintenance','numOfDoors','numOfPersons','luggage_boot','Safety','Target'])

# # 2(a) -------------------------------------- Shuffling the dataset
CarDataSet = CarDataSet.sample(frac=1, random_state = 42).reset_index(drop = True)
X_knn = CarDataSet.iloc[:, 0:6]
Y_knn = CarDataSet.iloc[:, [6]]

# # 2(b) -------------------------------------- Label encoder
X_knn = Assignment2.labelEncoder(X_knn.copy(),'Buying', ['low','med','high','vhigh'])
X_knn = Assignment2.labelEncoder(X_knn.copy(),'Maintenance', ['low','med','high','vhigh'])
X_knn = Assignment2.labelEncoder(X_knn.copy(),'numOfDoors', ['2','3','4','5more'])
X_knn = Assignment2.labelEncoder(X_knn.copy(),'numOfPersons', ['2','4','more'])
X_knn = Assignment2.labelEncoder(X_knn.copy(),'luggage_boot', ['small','med','big'])
X_knn = Assignment2.labelEncoder(X_knn.copy(),'Safety', ['low','med','high'])

Y_knn = Assignment2.labelEncoder(Y_knn.copy(),'Target', ['unacc','acc','good','vgood'])

# 2(a) -------------------------------------- 
# Split Training
X_trainKnn = X_knn.iloc[0:1000, :]
Y_trainKnn = Y_knn.iloc[0:1000, :]

# Split Validation
X_valKnn = X_knn.iloc[1000:1300, :].reset_index(drop=True)
Y_valKnn = Y_knn.iloc[1000:1300, :].reset_index(drop=True)

# Split Test
X_testKnn = X_knn.iloc[1300:, :].reset_index(drop=True)
Y_testKnn = Y_knn.iloc[1300:, :].reset_index(drop=True)

# 2(c) -------------------------------------- 
classifiers = [None]*11

y_predsTest = [None]*11
y_predsVals = [None]*11

Acc_OnTests = [None]*11
Acc_OnVals = [None]*11

X_trainKnns = [None]*11
Y_trainKnns = [None]*11

for i in range(1,11):
    X_trainKnns[i] = X_trainKnn.copy().iloc[0:((i)*100), :]
    Y_trainKnns[i] = Y_trainKnn.copy().iloc[0:((i)*100), :]
    
    classifiers[i] = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
    classifiers[i].fit(X_trainKnns[i],Y_trainKnns[i]["Target"].astype('int'))
    
    y_predsTest[i] = classifiers[i].predict(X_testKnn.astype('int'))
    y_predsVals[i] = classifiers[i].predict(X_valKnn.astype('int'))
    
    Acc_OnTests[i] = Assignment2.AccuracyTest(Y_testKnn.astype('int'), y_predsTest[i])
    Acc_OnVals[i] = Assignment2.AccuracyTest(Y_valKnn.astype('int'), y_predsVals[i])
    
    print('*******************************************************************************\n',' ', i, "- Acc_OnTests (Starting from 10% to 100%) : " , Acc_OnTests[i] , '\n',' ', i, "- Acc_OnVals (Starting from 10% to 100%) : ", Acc_OnVals[i])

Acc_OnTests.pop(0)
Acc_OnVals.pop(0)

plt.plot([10,20,30,40,50,60,70,80,90,100] ,Acc_OnTests, label = 'Test', c='#008037')
plt.plot([10,20,30,40,50,60,70,80,90,100] ,Acc_OnVals, label = 'Validation', c= '#67008c')
plt.scatter(x = [10,20,30,40,50,60,70,80,90,100], y = Acc_OnTests, c='#008037', marker = 'o', s=100)
plt.scatter(x = [10,20,30,40,50,60,70,80,90,100], y = Acc_OnVals, c='#67008c', marker = 'o', s=100)

plt.xlabel("portion of the training set")
plt.ylabel('Accuracy Score')
plt.title("Learning Curve")
plt.legend()
plt.show()

# 2(d) --------------------------------------
y_predsTest_k = [None]*11
y_predsVals_k = [None]*11

Acc_OnTests_k = [None]*11
Acc_OnVals_k = [None]*11

for i in range(1,11):
    classifier_K = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
    classifier_K.fit(X_trainKnn.astype('int'),Y_trainKnn["Target"].astype('int'))
    
    y_predsTest_k[i] = classifier_K.predict(X_testKnn.astype('int'))
    y_predsVals_k[i] = classifier_K.predict(X_valKnn.astype('int'))
    
    Acc_OnTests_k[i] = Assignment2.AccuracyTest(Y_testKnn.astype('int'), y_predsTest_k[i])
    Acc_OnVals_k[i] = Assignment2.AccuracyTest(Y_valKnn.astype('int'), y_predsVals_k[i])

Acc_OnVals_k.pop(0)
Acc_OnTests_k.pop(0)

# plt.plot(Acc_OnTests_k, label = 'Test', c='#008037')
plt.plot([1,2,3,4,5,6,7,8,9,10],Acc_OnVals_k, label = 'Validation', c = '#67008c')
# plt.scatter(x = [0,1,2,3,4,5,6,7,8,9], y = Acc_OnTests_k, c='#008037', marker = 'o', s=100)
plt.scatter(x = [1,2,3,4,5,6,7,8,9,10], y = Acc_OnVals_k, c='#67008c', marker = 'o', s=100)

plt.xlabel("Number of K")
plt.ylabel('Accuracy Score')
plt.title("Learning Curve")
plt.legend()
plt.show()

# 2(e) -------------------------------------- 
X_trainKnn_time10 = X_trainKnn.copy().iloc[0:100,:]
Y_trainKnn_time10 = Y_trainKnn.copy().iloc[0:100,:]

classifier_Time_2_full = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
classifier_Time_2_part = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)

classifier_Time_10_full = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
classifier_Time_10_part = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)

classifier_Time_2_full.fit(X_trainKnn.astype('int'),Y_trainKnn["Target"].astype('int'))
classifier_Time_2_part.fit(X_trainKnn_time10.astype('int'),Y_trainKnn_time10["Target"].astype('int'))
classifier_Time_10_full.fit(X_trainKnn.astype('int'),Y_trainKnn["Target"].astype('int'))
classifier_Time_10_part.fit(X_trainKnn_time10.astype('int'),Y_trainKnn_time10["Target"].astype('int'))

start2full = time.time()
y_predsTest_2_full = classifier_Time_2_full.predict(X_testKnn.astype('int'))
stop2full = time.time()
time2full = stop2full - start2full

start2Part = time.time()
y_predsTest_2_part = classifier_Time_2_part.predict(X_testKnn.astype('int'))
stop2Part = time.time()
time2Part = stop2Part - start2Part

start10full = time.time()
y_predsTest_10_full = classifier_Time_10_full.predict(X_testKnn.astype('int'))
stop10full = time.time()
time10full = stop10full - start10full

start10Part = time.time()
y_predsTest_10_part = classifier_Time_10_part.predict(X_testKnn.astype('int'))
stop10Part = time.time()
time10Part = stop10Part - start10Part

# Bar Chart
Bars = ['k = 2, 100%','k = 2, 10%','k = 10, 100%','k = 10, 10%']
Times = [time2full,time2Part,time10full,time10Part]
plt.bar(Bars, Times, 0.3, color ='#00BD7B')
plt.xlabel("4 cases")
plt.ylabel('Time')
plt.title("Prediction time on the testing set")