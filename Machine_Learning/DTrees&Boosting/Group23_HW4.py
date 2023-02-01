import pandas as pd, numpy as np, re
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from math import trunc
import sys
import subprocess
import time

try : 
    import xgboost as xgb
except:
    print("installing xgb ... ")
    subprocess.check_call([sys.executable, 'pip', 'install', 'xgboost'])


#------------------------------------Functions------------------------------------

# Function to read the Dataset
def readDataSet(DataSet_name, Sheet_Name):
    Extension = re.findall('((.csv)|(.xls)|(.xlsx))', DataSet_name)
    Extension = str(Extension)
    if '.csv' in Extension:
        Extension = '.csv'
    elif '.xls' in Extension:
        Extension = '.xls'
    elif '.xlsx' in Extension:
        Extension = '.xlsx'
    if Extension == ('.xls' or '.xlsx'):
        DataFrame = pd.read_excel(DataSet_name, sheet_name = Sheet_Name)
    elif Extension == '.csv':
        DataFrame = pd.read_csv(DataSet_name)
    return DataFrame
  
# SVM Classifier
def SVM(X ,Y ,GeneralizationTerm):
    ClassifierSVM = SVC(kernel="rbf", C = GeneralizationTerm, probability=True)
    ClassifierSVM.fit(X,Y)
    return ClassifierSVM

# to generate Confusion Matrix
def ConfusionMatrix(Y_Actual, Y_Pred):
    CF = confusion_matrix(Y_Actual, Y_Pred)
    return CF

# to Plot Confusion Matrix
def PLOT_ConfusionMatrix(CF,Title):
    sns.heatmap(CF, annot=True, fmt='d')
    plt.title(Title, fontsize = 15)
    plt.xlabel('Predicted', fontsize = 15)
    plt.ylabel('Actual', fontsize = 15)
    return plt.show()

# Accuracy on Test
def AccuracyTest(Y_Actual, Y_Pred):
    return accuracy_score(Y_Actual, Y_Pred) * 100

# DT Classifier
def DecisionTree_Gini(X ,Y):
    ClassifierDT = DecisionTreeClassifier()
    ClassifierDT.fit(X,Y)
    return ClassifierDT

# DT Classifier
def DecisionTree_Entropy(X ,Y):
    ClassifierDT = DecisionTreeClassifier(criterion='entropy')
    ClassifierDT.fit(X,Y)
    return ClassifierDT

def PlotDataPoints(numberOfClasses, ListOFClasses, XLabel, Ylabel ,labels ,S, Title):
    colorsOptions = ['#FF0000','#8941FF','blue','#00FF0F','#FF00AE','#000000','#0B9397','#1B49CD','#11AF29','#560078','#ED036A']
    MarkersOptions = ['h','*','v','^','D','x','X','P','H','d']
    colors = colorsOptions[0:numberOfClasses]
    Markers = MarkersOptions[0:numberOfClasses]
    
    for i in range(numberOfClasses):
        plt.scatter(x = ListOFClasses[i].iloc[:, 0:1], y = ListOFClasses[i].iloc[:, 1:2], c=colors[i], marker = Markers[i], s=S, label = labels[i])

    plt.xlabel(XLabel, fontsize = 15)
    plt.ylabel(Ylabel, fontsize = 15)
    plt.title(Title)
    plt.legend()
    return plt

# to separate the points that belong for each class
def GetListOfClasses(numberOfClasses, DataSet, TargetColumn):
        ls = [None] * numberOfClasses
        for i in range(0,numberOfClasses):
            ls[i] = DataSet.loc[DataSet[TargetColumn] == i]
        return ls

# T_Sne
def T_SNE(X):
    return TSNE(n_components=2).fit_transform(X)

# to plot the data 
def Plot(X,Y,Label,Color, Marker , S , Xlabel , Ylabel , Title):
    plt.plot(X, Y, label = Label, c = Color)
    plt.scatter(X,Y, c=Color, marker = Marker , s=S)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.legend()
    return plt

#-------------------------------------------------------------------Main-------------------------------------------------------------------
# 2(a) -------------------------------------- Read DataSet and Apply decision tree
Train = readDataSet('pendigits-tra.csv', 'pendigits-tra')
Train = Train.T.reset_index(drop=True).T
Test = readDataSet('pendigits-tes.csv', 'pendigits-tes')
Test = Test.T.reset_index(drop=True).T

XTrain = Train.iloc[:, :16]
YTrain = Train.iloc[:, 16]

XTest = Test.iloc[:, :16]
Ytest = Test.iloc[:, 16]

XTsne = T_SNE(XTest)

# Gini
Gini_Classifier = DecisionTree_Gini(XTrain ,YTrain)
Gini_Ypred = Gini_Classifier.predict(XTest)
Gini_report = classification_report(Ytest, Gini_Ypred)
Gini_cf = ConfusionMatrix(Ytest, Gini_Ypred)
PLOT_ConfusionMatrix(Gini_cf,'Decision Tree Gini')

XTsne = pd.concat([pd.DataFrame(XTsne), pd.DataFrame(Gini_Ypred)],axis=1 , ignore_index = True).astype(float)
GiniLs = GetListOfClasses(9, XTsne,  pd.DataFrame(XTsne).columns[2])
Labels = ['0','1','2','3','4','5','6','7','8']
PlotDataPoints(9, GiniLs, 'Component 0', 'Component 1' ,Labels ,5, 'Gini_Classifier').show()
XTsne = T_SNE(XTest)

# Entropy
Ent_Classifier = DecisionTree_Entropy(XTrain ,YTrain)
Ent_Ypred = Ent_Classifier.predict(XTest)
Ent_report = classification_report(Ytest, Ent_Ypred)
Ent_cf = ConfusionMatrix(Ytest, Ent_Ypred)
PLOT_ConfusionMatrix(Ent_cf,'Decision Tree Entropy')

XTsne = pd.concat([pd.DataFrame(XTsne), pd.DataFrame(Ent_Ypred)],axis=1 , ignore_index = True).astype(float)
EntLs = GetListOfClasses(9, XTsne,  pd.DataFrame(XTsne).columns[2])
PlotDataPoints(9, EntLs, 'Component 0', 'Component 1' ,Labels ,5, 'Entropy_Classifier').show()


# 3(a) -------------------------------------- #Bagging
#SVM
SVM_reports = []
for numOfEst in range(1, 3):
  SVM_estimator = BaggingClassifier(base_estimator=SVC(), n_estimators=numOfEst, random_state=0).fit(XTrain, YTrain)
  SVM_Ypred = SVM_estimator.predict(XTest)
  SVM_report = classification_report(Ytest, SVM_Ypred)
  SVM_reports.append(SVM_report)
  print('\t\tSVM Bagging: ',numOfEst, '\n',SVM_report)
  SVM_cf = ConfusionMatrix(Ytest, SVM_Ypred)
  PLOT_ConfusionMatrix(SVM_cf, f'SVM_Bagging {numOfEst}')
  
# Decision Tree
DT_reports = []
for numOfEst in range(1, 3):
  DT_estimator = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=numOfEst, random_state=0).fit(XTrain, YTrain)
  DT_Ypred = DT_estimator.predict(XTest)
  DT_report = classification_report(Ytest, DT_Ypred)
  DT_reports.append(DT_report)
  print('\t\tDecision Tree Bagging: ',numOfEst, '\n',DT_report)
  DT_cf = ConfusionMatrix(Ytest, DT_Ypred)
  PLOT_ConfusionMatrix(DT_cf, f'DT_Bagging {numOfEst}')


# 3(b) -------------------------------------- #Bagging
# Decision Tree
DT_Acc = []
nofEst = []
for numOfEst in range(10, 201,10):
  DT_estimator = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=numOfEst, random_state=0).fit(XTrain, YTrain)
  DT_Ypred = DT_estimator.predict(XTest)
  DT_Acc.append(AccuracyTest(Ytest, DT_Ypred))
  nofEst.append(numOfEst)

EstandAcc = pd.concat([pd.DataFrame(DT_Acc), pd.DataFrame(nofEst)],axis=1 , ignore_index = True).astype(float)
EstandAcc = EstandAcc.sort_values(by=[0],ascending=False)

Plot(nofEst,DT_Acc,'Best 20 Accuracies [range(10, 201,10)]','#009900', 'o' , 100 , 'Num of Estimators' , 'Accuracy' , 'Accuracy VS Num Of Estimators').show()

Plot(EstandAcc.iloc[:5,1],EstandAcc.iloc[:5,0],'Best 5 Accuracies','#009900', 'o' , 100 , 'Num of Estimators' , 'Accuracy' , 'Accuracy VS Num Of Estimators').show()


# 4(a) -------------------------------------- #Boosting
# Tuning the number of estimators Parameter
Boosting_Acc = []
numOfEst = []
for i in range(10,201,10):
   Boosting_estimator = GradientBoostingClassifier(n_estimators=i, random_state=0).fit(XTrain, YTrain)
   Boosting_Ypred = Boosting_estimator.predict(XTest)
   Boosting_Acc.append(AccuracyTest(Ytest, Boosting_Ypred))
   numOfEst.append(i)
   
Boosting_Est = pd.concat([pd.DataFrame(Boosting_Acc), pd.DataFrame(numOfEst)],axis=1 , ignore_index = True).astype(float)
Boosting_Est = Boosting_Est.sort_values(by=[0],ascending=False)
print(Boosting_Est.iloc[:4,1])

# Tuning learning rate parameter
Lr_rate = np.array([0.1,0.2,0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9])
Boosting_Acc = []
Lr = []
for i in Lr_rate:
   Boosting_estimator = GradientBoostingClassifier(learning_rate=i, random_state=0).fit(XTrain, YTrain)
   Boosting_Ypred = Boosting_estimator.predict(XTest)
   Boosting_Acc.append(AccuracyTest(Ytest, Boosting_Ypred))
   Lr.append(i)
   
Boosting_Lr = pd.concat([pd.DataFrame(Boosting_Acc), pd.DataFrame(Lr)],axis=1 , ignore_index = True).astype(float)
Boosting_Lr = Boosting_Lr.sort_values(by=[0],ascending=False)
print(Boosting_Lr.iloc[:4,1])

# Train GradientBoostingClassifier
Boosting_Acc = []
est = []
lr = []
estLS = list(Boosting_Est.iloc[:4,1])
LrLS = list(Boosting_Lr.iloc[:4,1])

for i in range(len(Boosting_Est.iloc[:4,1])):
    for j in range(len(Boosting_Lr.iloc[:4,1])):
       Boosting_estimator = GradientBoostingClassifier(n_estimators=trunc(estLS[i]),learning_rate=LrLS[j], random_state=0).fit(XTrain, YTrain)
       Boosting_Ypred = Boosting_estimator.predict(XTest)
       Boosting_Acc.append(AccuracyTest(Ytest, Boosting_Ypred))
       print('Done')
       est.append(trunc(estLS[i]))
       lr.append(LrLS[j])
   
Boosting = pd.concat([pd.DataFrame(Boosting_Acc), pd.DataFrame(est), pd.DataFrame(lr)],axis=1 , ignore_index = True).astype(float)
Boosting = Boosting.sort_values(by=[0],ascending=False)
print(Boosting.iloc[:4])

# Best 6 results
best_estLS = list(Boosting.iloc[:4,1].unique())
best_LrLS = list(Boosting.iloc[:4,2].unique())
GB_reports = []
GB_Acc = []
GBlr = []
GBEst = []

startGB = time.time()
for i in range(len(best_estLS)):
    for j in range(len(best_LrLS)):
       Boosting_estimator = GradientBoostingClassifier(n_estimators=trunc(best_estLS[i]),learning_rate=best_LrLS[j], random_state=0).fit(XTrain, YTrain)
       Boosting_Ypred = Boosting_estimator.predict(XTest)
       
       GB_report = classification_report(Ytest, Boosting_Ypred)
       GB_reports.append(DT_report)
       print('\t\tGradient Boosting Best 6 Accuracies:',"\tNum of Est = ",trunc(best_estLS[i]), '\tLearning Rate = ',best_LrLS[j], '\n',GB_report)
       GB_cf = ConfusionMatrix(Ytest, Boosting_Ypred)
       PLOT_ConfusionMatrix(GB_cf, f'Gradient Boosting {trunc(best_estLS[i]),best_LrLS[j]}')
       GB_Acc.append(AccuracyTest(Ytest, Boosting_Ypred))
       GBlr.append(best_LrLS[j])
       GBEst.append(trunc(best_estLS[i]))
       print('Done')
stopGB = time.time()
timeGB = stopGB - startGB
print(timeGB,'\n')

GB_eval = pd.concat([pd.DataFrame(GB_Acc), pd.DataFrame(GBEst), pd.DataFrame(GBlr)],axis=1 , ignore_index = True).astype(float)
GB_eval = GB_eval.sort_values(by=[0],ascending=False)
print(GB_eval)
              
# 4(b) and 4(c) -------------------------------------- #XGBoost
best_estLS = list(Boosting.iloc[:4,1].unique())
best_LrLS = list(Boosting.iloc[:4,2].unique())
XG_reports = []
XG_Acc = []
Xglr = []
XgEst = []

startXG = time.time()
for i in range(len(best_estLS)):
    for j in range(len(best_LrLS)):
        XG_estimator = xgb.XGBClassifier(objective = 'multi:softmax' ,num_class = 9, learning_rate = best_LrLS[j], n_estimators = trunc(best_estLS[i])).fit(XTrain,YTrain)
        XG_Ypred = XG_estimator.predict(XTest)
        XG_report = classification_report(Ytest, XG_Ypred)
        
        XG_reports.append(XG_report)
        print('\t\tXG_Boost Best 6 Accuracies:',"\tNum of Est = ",trunc(best_estLS[i]), '\tLearning Rate = ',best_LrLS[j], '\n',XG_report)
        XG_cf = ConfusionMatrix(Ytest, XG_Ypred)
        PLOT_ConfusionMatrix(XG_cf, f'XG_Boost {trunc(best_estLS[i]),best_LrLS[j]}')
        XG_Acc.append(AccuracyTest(Ytest, XG_Ypred))
        Xglr.append(best_LrLS[j])
        XgEst.append(trunc(best_estLS[i]))
        print('Done')
stopXG = time.time()
timeXG = stopXG - startXG
print(timeXG,'\n')

XG_eval = pd.concat([pd.DataFrame(XG_Acc), pd.DataFrame(XgEst), pd.DataFrame(Xglr)],axis=1 , ignore_index = True).astype(float)
XG_eval = XG_eval.sort_values(by=[0],ascending=False)
print(XG_eval)