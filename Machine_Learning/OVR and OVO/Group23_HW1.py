import pandas as pd, numpy as np,re
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Assignment1:
    DataFrame = pd.DataFrame()
    
    # Function to read the Dataset
    @staticmethod
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
            Assignment1.DataFrame = pd.read_excel(DataSet_name, sheet_name = Sheet_Name)
        elif Extension == '.csv':
            Assignment1.DataFrame = pd.read_csv(DataSet_name)
        return Assignment1.DataFrame
    
    # Function to Plot PairPlot
    @staticmethod
    def PairPLot(DataSet , TargetVariable):
        return plt.show(sns.pairplot(DataSet, hue = TargetVariable, palette="bright"))
    
    # Function to Encode Labels of Categorical data
    @staticmethod
    def labelEncoder(DataFrame , Categorical_Column, ListOfClassesNames):
        for i in range(len(ListOfClassesNames)):
            DataFrame.loc[DataFrame[str(Categorical_Column)] == ListOfClassesNames[i], str(Categorical_Column)] = i
        return DataFrame
    
    # Function to detemine which features will be eliminated
    @staticmethod
    def FeatureSelection(X,Y):
        Selector = ExtraTreesClassifier(n_estimators=2)
        Selector = Selector.fit(X, Y)
        return Selector.feature_importances_
    
    # SVM Classifier
    @staticmethod
    def SVM(X ,Y ,GeneralizationTerm):
        ClassifierSVM = SVC(kernel="rbf", C = GeneralizationTerm, probability=True)
        ClassifierSVM.fit(X,Y)
        return ClassifierSVM
    
    # PERCEP Classifier
    @staticmethod
    def PERCEP(X,Y,LearningRate,Epoch):
        ClassifierPERCEP = Perceptron(eta0=LearningRate, max_iter=Epoch)
        ClassifierPERCEP.fit(X,Y)
        return ClassifierPERCEP
    
    # to make Predictions
    @staticmethod
    def Pred(Classifier,X_Test):
        Predictions = Classifier.predict(X_Test)
        return Predictions
    
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
    
    # Slice the DataFrame based on The Different Classes (List Of DataFrames)
    @staticmethod
    def GetListOfClasses(numberOfClasses, DataSet, TargetColumn):
        ls = [None] * numberOfClasses
        for i in range(0,numberOfClasses):
            ls[i] = DataSet.loc[DataSet[TargetColumn] == i]
        return ls
    
    # Plot the Different Classes
    @staticmethod
    def PlotData(numberOfClasses, ListOFClasses):
        colorsOptions = ['#FF00AE','blue','#8941FF','#00FF0F','#FF0000','#000000','#0B9397','#1B49CD','#11AF29','#560078','#ED036A']
        MarkersOptions = ['*','^','v','P','D','x','X','h','H','d']
        colors = colorsOptions[0:numberOfClasses]
        Markers = MarkersOptions[0:numberOfClasses]
        if ((numberOfClasses == 2) and (len(ListOFClasses[1].LPR.copy().reset_index(drop=True).values.tolist())+len(ListOFClasses[0].LPR.copy().reset_index(drop=True).values.tolist())) == 80):
            for i in range(0,numberOfClasses):
                for j in range(len(ListOFClasses[i])):
                    plt.annotate('Rest' if i==0 else '' , (ListOFClasses[i].LPR.copy().reset_index(drop=True).values.tolist()[j], ListOFClasses[i].PEG.copy().reset_index(drop=True).values.tolist()[j]))
                plt.scatter(x = ListOFClasses[i].iloc[:, 0:1], y = ListOFClasses[i].iloc[:, 1:2], c=colors[i], marker = Markers[i], s=50)
        elif numberOfClasses == 4:
            for i in range(0,numberOfClasses):
                for j in range(len(ListOFClasses[i])):
                    plt.annotate('Very Low' if i==0 else 'Low' if i==1 else 'Medium' if i==2 else 'High' if i==3 else None, (ListOFClasses[i].LPR.copy().reset_index(drop=True).values.tolist()[j], ListOFClasses[i].PEG.copy().reset_index(drop=True).values.tolist()[j]))
                plt.scatter(x = ListOFClasses[i].iloc[:, 0:1], y = ListOFClasses[i].iloc[:, 1:2], c=colors[i], marker = Markers[i], s=50)
        plt.xlabel('LPR', fontsize = 15)
        plt.ylabel('PEG', fontsize = 15)
        return plt.show()
    
    # Plot the Different Classes OVR
    @staticmethod
    def WrongPointsOvR(Y_actual, Y_Pred , Xtest,FristClass):
        TruePoints = Xtest[Y_actual == Y_Pred]
        FalsePoints = Xtest[Y_actual != Y_Pred] 
        plt.scatter(x = TruePoints.iloc[:,0], y = TruePoints.iloc[:,1], c='blue', s=50 , label = FristClass)
        plt.scatter(x = FalsePoints.iloc[:,0], y =FalsePoints.iloc[:,1], c='red', marker ="X" , s=100, label = 'Wrong')
        plt.legend()
        plt.xlabel('LPR')
        plt.ylabel('PEG')
        plt
        
    # Plot the Different Classes OVO
    @staticmethod
    def WrongPointsOvO(Y_actual, Y_Pred , Xtest):
        TruePoints = Xtest[Y_actual == Y_Pred]
        FalsePoints = Xtest[Y_actual != Y_Pred] 
        plt.scatter(x = TruePoints.iloc[:,0], y = TruePoints.iloc[:,1], c='blue', s=50, label = 'TwoClasses')
        plt.scatter(x = FalsePoints.iloc[:,0], y =FalsePoints.iloc[:,1], c='red', marker ="X" , s=100, label = 'Wrong')
        plt.legend()
        plt.xlabel('LPR')
        plt.ylabel('PEG')
        plt
    
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
    
    # One Vs Rest Spliting
    @staticmethod
    def OvR(DataSet,RequiredClass, Categorical_Column):
        BinaryDF = DataSet.copy()
        if RequiredClass == 0:
            BinaryDF.loc[BinaryDF[str(Categorical_Column)] != RequiredClass, str(Categorical_Column)] = 2
            BinaryDF.loc[BinaryDF[str(Categorical_Column)] == RequiredClass, str(Categorical_Column)] = 1
            BinaryDF.loc[BinaryDF[str(Categorical_Column)] == 2, str(Categorical_Column)] = 0
        else:
            BinaryDF.loc[BinaryDF[str(Categorical_Column)] != RequiredClass, str(Categorical_Column)] = 0
            BinaryDF.loc[BinaryDF[str(Categorical_Column)] == RequiredClass, str(Categorical_Column)] = 1
        return BinaryDF
    
    # One Vs Rest Spliting
    @staticmethod
    def OvO(FristClass, SecondClass, DataSet, Categorical_Column):
        TwoClassesDF = DataSet.copy()
        TwoClassesDF = TwoClassesDF[(TwoClassesDF[str(Categorical_Column)] == FristClass) | (TwoClassesDF[str(Categorical_Column)] == SecondClass)]
        TwoClassesDF = TwoClassesDF.reset_index(drop=True)
        TwoClassesDF.loc[TwoClassesDF[str(Categorical_Column)] == FristClass, str(Categorical_Column)] = FristClass
        TwoClassesDF.loc[TwoClassesDF[str(Categorical_Column)] == SecondClass, str(Categorical_Column)] = SecondClass
        return TwoClassesDF

#----------------------------------------MAIN----------------------------------------
# labelEncoder ---> Very Low = 0, Low = 1, Medium = 2 ,High = 3

# Excel files and sheets names
file1 = 'DUMD_train.csv'
file2 = 'DUMD_test.csv'

Sheet_Name1 = 'DUMD_train'
Sheet_Name2 = 'DUMD_test'

# Read DataSet 
OBJ = Assignment1()
TrainDataSet = OBJ.readDataSet(file1, Sheet_Name1)
TestDataSet = OBJ.readDataSet(file2, Sheet_Name2)
DataSetTrain = TrainDataSet
DataSetTest = TestDataSet

# Classes Names
ClassesNames = ['Very Low','Low','Medium','High']

# labelEncoder ---> Very Low = 0 , High = 3
TrainDataSet = Assignment1.labelEncoder(TrainDataSet,'UNS', ClassesNames)
TestDataSet = Assignment1.labelEncoder(TestDataSet,'UNS', ClassesNames)

# Split Train and Test
TrainY = TrainDataSet.iloc[:,5:6].squeeze().astype('int')
TestY = TestDataSet.iloc[:,5:6].squeeze().astype('int')
TrainDataSet = TrainDataSet.iloc[:, 0:5]
TestDataSet = TestDataSet.iloc[:, 0:5]

# Select two Features (Print the feature importance score) and plot pairplot
Assignment1.PairPLot(DataSetTrain , 'UNS')
print(Assignment1.FeatureSelection(TrainDataSet,TrainY.squeeze().astype('int')))
      
# Select LPR and PEG Features --> they have the highest feature importance score and they is the most seperable 2 features based on pairplot
TrainX = TrainDataSet.iloc[:, 3:6]
TestX = TestDataSet.iloc[:, 3:6]
DataSetTrain = DataSetTrain.iloc[:,3:]
DataSetTest = DataSetTest.iloc[:,3:]
del TrainDataSet, TestDataSet, file1, file2, Sheet_Name1, Sheet_Name2

# PairPlot For the selected two features
Assignment1.PairPLot(DataSetTrain, 'UNS')

# For Training we will use ----> TrainX & TrainY
# For Testing we will use ----> TestX & TestY

# Linear SVM & PERCEP Classifier
ClassifierSVM = Assignment1.SVM(TrainX, TrainY, 100)
ClassifierPERCEP = Assignment1.PERCEP(TrainX, TrainY, 0.1, 300)

# Use Classifiers to make predictions
Y_PredSVM = Assignment1.Pred(ClassifierSVM,TestX)
Y_PredPERCEP = Assignment1.Pred(ClassifierPERCEP,TestX)

# Accuracy on Train SVM and PERCEP
AccSVM_OnTrain = Assignment1.AccuracyTrain(ClassifierSVM, TrainX, TrainY)
AccPERCEP_OnTrain = Assignment1.AccuracyTrain(ClassifierPERCEP, TrainX, TrainY)

# Accuracy on Test SVM and PERCEP
AccSVM_OnTest = Assignment1.AccuracyTest(TestY, Y_PredSVM)
AccPERCEP_OnTest = Assignment1.AccuracyTest(TestY, Y_PredPERCEP)

# Confusion Matrix
CF_MatrixSVM = Assignment1.ConfusionMatrix(TestY, Y_PredSVM)
CF_MatrixPERCEP = Assignment1.ConfusionMatrix(TestY, Y_PredPERCEP)

# Plot Confusion Matrix
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM,'SVM') # SVM
Assignment1.PLOT_ConfusionMatrix(CF_MatrixPERCEP, 'PERCEP') # PERCEP

# Create list of 4 DataFrames (DataFrame for each class)
ListOfClasses = Assignment1.GetListOfClasses(4, DataSetTest, 'UNS')

# Plot the data and the Decesion Boundaries
Assignment1.Boundaries(TestX,ClassifierSVM,'SVM',4)
Assignment1.PlotData(4,ListOfClasses)

Assignment1.Boundaries(TestX,ClassifierPERCEP,'PERCEP',4)
Assignment1.PlotData(4,ListOfClasses)

# One Vs Rest (binarized labels)
OvR_DF_Train = Assignment1.OvR(DataSetTrain.copy(),0,'UNS')
OvR_DF_0_TrainY = OvR_DF_Train.iloc[:, 2:].squeeze().astype('int')
OvR_DF_Train = OvR_DF_Train.iloc[:, 0:2]

OvR_DF_Test = Assignment1.OvR(DataSetTest.copy(),0,'UNS')
OvR_DF_0_TestY = OvR_DF_Test.iloc[:, 2:].squeeze().astype('int')
OvR_DF_Test = OvR_DF_Test.iloc[:, 0:2]

###########################################################################

OvR_DF_Train = Assignment1.OvR(DataSetTrain.copy(),1,'UNS')
OvR_DF_1_TrainY = OvR_DF_Train.iloc[:, 2:].squeeze().astype('int')
OvR_DF_Train = OvR_DF_Train.iloc[:, 0:2]

OvR_DF_Test = Assignment1.OvR(DataSetTest.copy(),1,'UNS')
OvR_DF_1_TestY = OvR_DF_Test.iloc[:, 2:].squeeze().astype('int')
OvR_DF_Test = OvR_DF_Test.iloc[:, 0:2]

###########################################################################

OvR_DF_Train = Assignment1.OvR(DataSetTrain.copy(),2,'UNS')
OvR_DF_2_TrainY = OvR_DF_Train.iloc[:, 2:].squeeze().astype('int')
OvR_DF_Train = OvR_DF_Train.iloc[:, 0:2]

OvR_DF_Test = Assignment1.OvR(DataSetTest.copy(),2,'UNS')
OvR_DF_2_TestY = OvR_DF_Test.iloc[:, 2:].squeeze().astype('int')
OvR_DF_Test = OvR_DF_Test.iloc[:, 0:2]

###########################################################################

OvR_DF_Train = Assignment1.OvR(DataSetTrain.copy(),3,'UNS')
OvR_DF_3_TrainY = OvR_DF_Train.iloc[:, 2:].squeeze().astype('int')
OvR_DF_Train = OvR_DF_Train.iloc[:, 0:2]

OvR_DF_Test = Assignment1.OvR(DataSetTest.copy(),3,'UNS')
OvR_DF_3_TestY = OvR_DF_Test.iloc[:, 2:].squeeze().astype('int')
OvR_DF_Test = OvR_DF_Test.iloc[:, 0:2]

###########################################################################

# Build 4 SVM Models for each Binary Classification proplem
ClassifierSVM_0 = Assignment1.SVM(OvR_DF_Train, OvR_DF_0_TrainY, 200)
ClassifierSVM_1 = Assignment1.SVM(OvR_DF_Train, OvR_DF_1_TrainY, 200)
ClassifierSVM_2 = Assignment1.SVM(OvR_DF_Train, OvR_DF_2_TrainY, 200)
ClassifierSVM_3 = Assignment1.SVM(OvR_DF_Train, OvR_DF_3_TrainY, 200)

# Use Classifiers to make predictions
Y_PredSVM_0 = Assignment1.Pred(ClassifierSVM_0, OvR_DF_Test)
Y_PredSVM_0_Prop = ClassifierSVM_0.predict_proba(OvR_DF_Test)[:,1].reshape(-1,1)

Y_PredSVM_1 = Assignment1.Pred(ClassifierSVM_1, OvR_DF_Test)
Y_PredSVM_1_Prop = ClassifierSVM_1.predict_proba(OvR_DF_Test)[:,1].reshape(-1,1)

Y_PredSVM_2 = Assignment1.Pred(ClassifierSVM_2, OvR_DF_Test)
Y_PredSVM_2_Prop = ClassifierSVM_2.predict_proba(OvR_DF_Test)[:,1].reshape(-1,1)

Y_PredSVM_3 = Assignment1.Pred(ClassifierSVM_3, OvR_DF_Test)
Y_PredSVM_3_Prop = ClassifierSVM_3.predict_proba(OvR_DF_Test)[:,1].reshape(-1,1)

# Accuracy on Train SVM
AccSVM_OnTrain_0 = Assignment1.AccuracyTrain(ClassifierSVM_0, OvR_DF_Train, OvR_DF_0_TrainY)
AccSVM_OnTrain_1 = Assignment1.AccuracyTrain(ClassifierSVM_1, OvR_DF_Train, OvR_DF_1_TrainY)
AccSVM_OnTrain_2 = Assignment1.AccuracyTrain(ClassifierSVM_2, OvR_DF_Train, OvR_DF_2_TrainY)
AccSVM_OnTrain_3 = Assignment1.AccuracyTrain(ClassifierSVM_3, OvR_DF_Train, OvR_DF_3_TrainY)

# Accuracy on Test SVM
AccSVM_OnTest_0 = Assignment1.AccuracyTest(OvR_DF_0_TestY, Y_PredSVM_0)
AccSVM_OnTest_1 = Assignment1.AccuracyTest(OvR_DF_1_TestY, Y_PredSVM_1)
AccSVM_OnTest_2 = Assignment1.AccuracyTest(OvR_DF_2_TestY, Y_PredSVM_2)
AccSVM_OnTest_3 = Assignment1.AccuracyTest(OvR_DF_3_TestY, Y_PredSVM_3)

# Plot Confusion Matrix
CF_MatrixSVM_0 = Assignment1.ConfusionMatrix(OvR_DF_0_TestY, Y_PredSVM_0)
CF_MatrixSVM_1 = Assignment1.ConfusionMatrix(OvR_DF_1_TestY, Y_PredSVM_1)
CF_MatrixSVM_2 = Assignment1.ConfusionMatrix(OvR_DF_2_TestY, Y_PredSVM_2)
CF_MatrixSVM_3 = Assignment1.ConfusionMatrix(OvR_DF_3_TestY, Y_PredSVM_3)

Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_0,'OVR_SVM_0 (Very Low Vs Rest)')
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_1,'OVR_SVM_1 (Low Vs Rest)')
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_2,'OVR_SVM_2 (Medium Vs Rest)')
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_3,'OVR_SVM_3 (High Vs Rest)')

# Create list of 2 DataFrames of each class (DataFrame for each category)
ListOfClasses_0 = Assignment1.GetListOfClasses(2, pd.concat([OvR_DF_Test, OvR_DF_0_TestY], axis=1, join='inner'),'UNS')
ListOfClasses_1 = Assignment1.GetListOfClasses(2, pd.concat([OvR_DF_Test, OvR_DF_1_TestY], axis=1, join='inner'),'UNS')
ListOfClasses_2 = Assignment1.GetListOfClasses(2, pd.concat([OvR_DF_Test, OvR_DF_2_TestY], axis=1, join='inner'),'UNS')
ListOfClasses_3 = Assignment1.GetListOfClasses(2, pd.concat([OvR_DF_Test, OvR_DF_3_TestY], axis=1, join='inner'),'UNS')

# Plot the data and the Decesion Boundaries
Assignment1.Boundaries(OvR_DF_Test, ClassifierSVM_0,'OVR_SVM_0 (Very Low Vs Rest)',2)
Assignment1.WrongPointsOvR(OvR_DF_0_TestY, Y_PredSVM_0, OvR_DF_Test, 'Very Low')
Assignment1.PlotData(2,ListOfClasses_0)

Assignment1.Boundaries(OvR_DF_Test, ClassifierSVM_1,'OVR_SVM_1 (Low Vs Rest)',2)
Assignment1.WrongPointsOvR(OvR_DF_1_TestY, Y_PredSVM_1, OvR_DF_Test , 'Low')
Assignment1.PlotData(2,ListOfClasses_1)

Assignment1.Boundaries(OvR_DF_Test, ClassifierSVM_2,'OVR_SVM_2 (Medium Vs Rest)',2)
Assignment1.WrongPointsOvR(OvR_DF_2_TestY, Y_PredSVM_2, OvR_DF_Test, 'Medium')
Assignment1.PlotData(2,ListOfClasses_2)

Assignment1.Boundaries(OvR_DF_Test ,ClassifierSVM_3,'OVR_SVM_3 (High Vs Rest)',2)
Assignment1.WrongPointsOvR(OvR_DF_3_TestY, Y_PredSVM_3, OvR_DF_Test,'High')
Assignment1.PlotData(2,ListOfClasses_3)

# Aggregation OvR
YProbALL_OvR = np.argmax(np.hstack((Y_PredSVM_0_Prop, Y_PredSVM_1_Prop, Y_PredSVM_2_Prop, Y_PredSVM_3_Prop)), axis=1)
AccSVM_OnTest_OvR_All = Assignment1.AccuracyTest(TestY, YProbALL_OvR)
CF_MatrixSVM_OvR_All = Assignment1.ConfusionMatrix(TestY, YProbALL_OvR)
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_OvR_All,'OVR_SVM_ALL')

#--------------------------------------------OvO--------------------------------------------

# One Vs One (binarized labels)
OvO_DF_0_Train = Assignment1.OvO(0, 1, DataSetTrain, 'UNS')
OvO_DF_0_TrainX = OvO_DF_0_Train.iloc[:, 0:2]
OvO_DF_0_TrainY = OvO_DF_0_Train.iloc[:, 2:].squeeze().astype('int')

#############################################################################

OvO_DF_1_Train = Assignment1.OvO(0, 2, DataSetTrain, 'UNS')
OvO_DF_1_TrainX = OvO_DF_1_Train.iloc[:, 0:2]
OvO_DF_1_TrainY = OvO_DF_1_Train.iloc[:, 2:].squeeze().astype('int')

#############################################################################

OvO_DF_2_Train = Assignment1.OvO(0, 3, DataSetTrain, 'UNS')
OvO_DF_2_TrainX = OvO_DF_2_Train.iloc[:, 0:2]
OvO_DF_2_TrainY = OvO_DF_2_Train.iloc[:, 2:].squeeze().astype('int')

#############################################################################

OvO_DF_3_Train = Assignment1.OvO(1, 2, DataSetTrain, 'UNS')
OvO_DF_3_TrainX = OvO_DF_3_Train.iloc[:, 0:2]
OvO_DF_3_TrainY = OvO_DF_3_Train.iloc[:, 2:].squeeze().astype('int')

#############################################################################

OvO_DF_4_Train = Assignment1.OvO(1, 3, DataSetTrain, 'UNS')
OvO_DF_4_TrainX = OvO_DF_4_Train.iloc[:, 0:2]
OvO_DF_4_TrainY = OvO_DF_4_Train.iloc[:, 2:].squeeze().astype('int')

#############################################################################

OvO_DF_5_Train = Assignment1.OvO(2, 3, DataSetTrain, 'UNS')
OvO_DF_5_TrainX = OvO_DF_5_Train.iloc[:, 0:2]
OvO_DF_5_TrainY = OvO_DF_5_Train.iloc[:, 2:].squeeze().astype('int')

# Build 6 SVM Models for each Binary Classification proplem
ClassifierSVM_OVO_0 = Assignment1.SVM(OvO_DF_0_TrainX, OvO_DF_0_TrainY, 200)
ClassifierSVM_OVO_1 = Assignment1.SVM(OvO_DF_1_TrainX, OvO_DF_1_TrainY, 200)
ClassifierSVM_OVO_2 = Assignment1.SVM(OvO_DF_2_TrainX, OvO_DF_2_TrainY, 200)
ClassifierSVM_OVO_3 = Assignment1.SVM(OvO_DF_3_TrainX, OvO_DF_3_TrainY, 200)
ClassifierSVM_OVO_4 = Assignment1.SVM(OvO_DF_4_TrainX, OvO_DF_4_TrainY, 200)
ClassifierSVM_OVO_5 = Assignment1.SVM(OvO_DF_5_TrainX, OvO_DF_5_TrainY, 200)

# Use Classifiers to make predictions
Y_PredSVM_OVO_0 = Assignment1.Pred(ClassifierSVM_OVO_0, TestX)
Y_PredSVM_0_Prop_OVO = ClassifierSVM_OVO_0.predict_proba(TestX)

Y_PredSVM_OVO_1 = Assignment1.Pred(ClassifierSVM_OVO_1, TestX)
Y_PredSVM_1_Prop_OVO = ClassifierSVM_OVO_1.predict_proba(TestX)

Y_PredSVM_OVO_2 = Assignment1.Pred(ClassifierSVM_OVO_2, TestX)
Y_PredSVM_2_Prop_OVO = ClassifierSVM_OVO_2.predict_proba(TestX)

Y_PredSVM_OVO_3 = Assignment1.Pred(ClassifierSVM_OVO_3, TestX)
Y_PredSVM_3_Prop_OVO = ClassifierSVM_OVO_3.predict_proba(TestX)

Y_PredSVM_OVO_4 = Assignment1.Pred(ClassifierSVM_OVO_4, TestX)
Y_PredSVM_4_Prop_OVO = ClassifierSVM_OVO_4.predict_proba(TestX)

Y_PredSVM_OVO_5 = Assignment1.Pred(ClassifierSVM_OVO_5, TestX)
Y_PredSVM_5_Prop_OVO = ClassifierSVM_OVO_5.predict_proba(TestX)

# Accuracy on Train SVM
AccSVM_OnTrain_OVO_0 = Assignment1.AccuracyTrain(ClassifierSVM_OVO_0, OvO_DF_0_TrainX, OvO_DF_0_TrainY)
AccSVM_OnTrain_OVO_1 = Assignment1.AccuracyTrain(ClassifierSVM_OVO_1, OvO_DF_1_TrainX, OvO_DF_1_TrainY)
AccSVM_OnTrain_OVO_2 = Assignment1.AccuracyTrain(ClassifierSVM_OVO_2, OvO_DF_2_TrainX, OvO_DF_2_TrainY)
AccSVM_OnTrain_OVO_3 = Assignment1.AccuracyTrain(ClassifierSVM_OVO_3, OvO_DF_3_TrainX, OvO_DF_3_TrainY)
AccSVM_OnTrain_OVO_4 = Assignment1.AccuracyTrain(ClassifierSVM_OVO_4, OvO_DF_4_TrainX, OvO_DF_4_TrainY)
AccSVM_OnTrain_OVO_5 = Assignment1.AccuracyTrain(ClassifierSVM_OVO_5, OvO_DF_5_TrainX, OvO_DF_5_TrainY)

# Accuracy on Test SVM
AccSVM_OnTest_OVO_0 = Assignment1.AccuracyTest(TestY, Y_PredSVM_OVO_0)
AccSVM_OnTest_OVO_1 = Assignment1.AccuracyTest(TestY, Y_PredSVM_OVO_1)
AccSVM_OnTest_OVO_2 = Assignment1.AccuracyTest(TestY, Y_PredSVM_OVO_2)
AccSVM_OnTest_OVO_3 = Assignment1.AccuracyTest(TestY, Y_PredSVM_OVO_3)
AccSVM_OnTest_OVO_4 = Assignment1.AccuracyTest(TestY, Y_PredSVM_OVO_4)
AccSVM_OnTest_OVO_5 = Assignment1.AccuracyTest(TestY, Y_PredSVM_OVO_5)

# Plot Confusion Matrix
CF_MatrixSVM_OVO_0 = Assignment1.ConfusionMatrix(TestY, Y_PredSVM_OVO_0)
CF_MatrixSVM_OVO_1 = Assignment1.ConfusionMatrix(TestY, Y_PredSVM_OVO_1)
CF_MatrixSVM_OVO_2 = Assignment1.ConfusionMatrix(TestY, Y_PredSVM_OVO_2)
CF_MatrixSVM_OVO_3 = Assignment1.ConfusionMatrix(TestY, Y_PredSVM_OVO_3)
CF_MatrixSVM_OVO_4 = Assignment1.ConfusionMatrix(TestY, Y_PredSVM_OVO_4)
CF_MatrixSVM_OVO_5 = Assignment1.ConfusionMatrix(TestY, Y_PredSVM_OVO_5)

Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_OVO_0,'OVO_SVM_0 (Very Low Vs Low)')
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_OVO_1,'OVO_SVM_1 (Very Low Vs Medium)')
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_OVO_2,'OVO_SVM_2 (Very Low Vs High)')
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_OVO_3,'OVO_SVM_3 (Low Vs Medium)')
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_OVO_4,'OVO_SVM_4 (Low Vs High)')
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_OVO_5,'OVO_SVM_5 (Medium Vs High)')

# Create list of 2 DataFrames of each class (DataFrame for each category)
ListOfClasses_OVO = Assignment1.GetListOfClasses(2, DataSetTest,'UNS')

# Plot the data and the Decesion 
Assignment1.Boundaries(TestX, ClassifierSVM_OVO_0,'OVO_SVM_0 (Very Low Vs Low)',2)
Assignment1.WrongPointsOvO(TestY, Y_PredSVM_OVO_0, TestX)
Assignment1.PlotData(2,ListOfClasses_OVO)

Assignment1.Boundaries(TestX, ClassifierSVM_OVO_1,'OVO_SVM_1 (Very Low Vs Medium)',2)
Assignment1.WrongPointsOvO(TestY, Y_PredSVM_OVO_1, TestX)
Assignment1.PlotData(2,ListOfClasses_OVO)

Assignment1.Boundaries(TestX, ClassifierSVM_OVO_2,'OVO_SVM_2 (Very Low Vs High)',2)
Assignment1.WrongPointsOvO(TestY, Y_PredSVM_OVO_2, TestX)
Assignment1.PlotData(2,ListOfClasses_OVO)

Assignment1.Boundaries(TestX, ClassifierSVM_OVO_3,'OVO_SVM_3 (Low Vs Medium)',2)
Assignment1.WrongPointsOvO(TestY, Y_PredSVM_OVO_3, TestX)
Assignment1.PlotData(2,ListOfClasses_OVO)

Assignment1.Boundaries(TestX, ClassifierSVM_OVO_4,'OVO_SVM_4 (Low Vs High)',2)
Assignment1.WrongPointsOvO(TestY, Y_PredSVM_OVO_4, TestX)
Assignment1.PlotData(2,ListOfClasses_OVO)

Assignment1.Boundaries(TestX, ClassifierSVM_OVO_5,'OVO_SVM_5 (Medium Vs High)',2)
Assignment1.WrongPointsOvO(TestY, Y_PredSVM_OVO_5, TestX)
Assignment1.PlotData(2,ListOfClasses_OVO)

# Aggregation OvO
VeryLow_Aggregation = (Y_PredSVM_0_Prop_OVO[: , 0] + Y_PredSVM_1_Prop_OVO[: , 0] + Y_PredSVM_2_Prop_OVO[: , 0])/3
Low_Aggregation = (Y_PredSVM_0_Prop_OVO[: , 1] + Y_PredSVM_3_Prop_OVO[: , 0] + Y_PredSVM_4_Prop_OVO[: , 0])/3
Medium_Aggregation = (Y_PredSVM_1_Prop_OVO[: , 1] + Y_PredSVM_3_Prop_OVO[: , 1] + Y_PredSVM_5_Prop_OVO[: , 0])/3
High_Aggregation = (Y_PredSVM_2_Prop_OVO[: , 1] + Y_PredSVM_4_Prop_OVO[: , 1] + Y_PredSVM_5_Prop_OVO[: , 1])/3

YProbALL_OvO = np.argmax(np.hstack((VeryLow_Aggregation.reshape(-1,1), Low_Aggregation.reshape(-1,1), Medium_Aggregation.reshape(-1,1), High_Aggregation.reshape(-1,1))), axis=1)
AccSVM_OnTest_OvO_All = Assignment1.AccuracyTest(TestY, YProbALL_OvO)
CF_MatrixSVM_OvO_All = Assignment1.ConfusionMatrix(TestY, YProbALL_OvO)
Assignment1.PLOT_ConfusionMatrix(CF_MatrixSVM_OvO_All,'OVO_SVM_ALL')