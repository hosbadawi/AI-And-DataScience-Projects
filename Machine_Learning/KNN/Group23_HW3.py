import pandas as pd, numpy as np,re
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.utils.multiclass import unique_labels
from sklearn_som.som import SOM
from sklearn.cluster import DBSCAN
from minisom import MiniSom


class Assignment3:
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
            Assignment3.DataFrame = pd.read_excel(DataSet_name, sheet_name = Sheet_Name)
        elif Extension == '.csv':
            Assignment3.DataFrame = pd.read_csv(DataSet_name)
        return Assignment3.DataFrame
    
    @staticmethod
    def Split(X,Y, TestSize , random=0):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=TestSize, random_state=random)
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        return x_train, x_test, y_train, y_test
    
    @staticmethod
    def KNN(x_train,x_test,y_train,y_test,n):
        model = KNeighborsClassifier(n_neighbors=n)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report  = classification_report(y_test, y_pred)
        return report , y_pred, model
    
    @staticmethod
    def Logistic(x_train,x_test,y_train,y_test):
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report  = classification_report(y_test, y_pred)
        return report , y_pred , model
    
    @staticmethod
    def Plot(X,Y,Label,Color, Marker , S , Xlabel , Ylabel , Title):
        plt.plot(X, Y, label = Label, c = Color)
        plt.scatter(X,Y, c=Color, marker = Marker , s=S)
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.title(Title)
        plt.legend()
        plt.show()
        return plt
    
    # Accuracy on Test
    @staticmethod
    def AccuracyTest(Y_Actual, Y_Pred):
        return accuracy_score(Y_Actual, Y_Pred) * 100
    
    # Accuracy on Test
    @staticmethod
    def T_SNE(X):
        return TSNE(n_components=2).fit_transform(X)
    
    # Slice the DataFrame based on The Different Classes (List Of DataFrames)
    @staticmethod
    def GetListOfClasses(numberOfClasses, DataSet, TargetColumn):
        ls = [None] * numberOfClasses
        for i in range(0,numberOfClasses):
            ls[i] = DataSet.loc[DataSet[TargetColumn] == i]
        return ls
    
    @staticmethod
    def PlotDataPoints(numberOfClasses, ListOFClasses, XLabel, Ylabel , Label0,label1 , S, Title):
        colorsOptions = ['#FF0000','#8941FF','blue','#00FF0F',' #FF00AE','#000000','#0B9397','#1B49CD','#11AF29','#560078','#ED036A']
        MarkersOptions = ['h','*','v','^','D','x','X','P','H','d']
        colors = colorsOptions[0:numberOfClasses]
        Markers = MarkersOptions[0:numberOfClasses]
        
        plt.scatter(x = ListOFClasses[0].iloc[:, 0:1], y = ListOFClasses[0].iloc[:, 1:2], c=colors[0], marker = Markers[0], s=S, label = Label0)
        plt.scatter(x = ListOFClasses[1].iloc[:, 0:1], y = ListOFClasses[1].iloc[:, 1:2], c=colors[1], marker = Markers[1], s=S, label = label1)

        plt.xlabel(XLabel, fontsize = 15)
        plt.ylabel(Ylabel, fontsize = 15)
        plt.title(Title)
        plt.legend()
        return plt
    
    @staticmethod
    def PlotDataPoints3(numberOfClasses, ListOFClasses, XLabel, Ylabel , Label0,label1 , label2 , S, Title):
        colorsOptions = ['#FF0000','#8941FF','blue','#00FF0F',' #FF00AE','#000000','#0B9397','#1B49CD','#11AF29','#560078','#ED036A']
        MarkersOptions = ['h','*','v','^','D','x','X','P','H','d']
        colors = colorsOptions[0:numberOfClasses]
        Markers = MarkersOptions[0:numberOfClasses]
        
        plt.scatter(x = ListOFClasses[0].iloc[:, 0:1], y = ListOFClasses[0].iloc[:, 1:2], c=colors[0], marker = Markers[0], s=S, label = Label0)
        plt.scatter(x = ListOFClasses[1].iloc[:, 0:1], y = ListOFClasses[1].iloc[:, 1:2], c=colors[1], marker = Markers[1], s=S, label = label1)
        plt.scatter(x = ListOFClasses[2].iloc[:, 0:1], y = ListOFClasses[2].iloc[:, 1:2], c=colors[2], marker = Markers[2], s=S, label = label2)

        
        plt.xlabel(XLabel, fontsize = 15)
        plt.ylabel(Ylabel, fontsize = 15)
        plt.title(Title)
        plt.legend()
        return plt
    
    @staticmethod
    def Kmeans(X, n, Random):
        kmeans = KMeans(n_clusters=n, random_state=Random).fit(X)
        score = silhouette_score(X, kmeans.labels_, metric='euclidean')
        return score
    
    @staticmethod
    def Pca(x_train, x_test, n):
        pca = PCA(n_components = n,  random_state=0)
        X_train = pca.fit_transform(x_train)
        X_test = pca.transform(x_test)
        return X_train, X_test
    
    @staticmethod
    def Bar(x, y, W, lsOFcolor , Xlabel, Ylabel , Title, Label):
        plt.bar(x, y, W, color = lsOFcolor, label=Label)
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.legend(loc = 3)
        plt.title(Title)
        return plt
    
    @staticmethod   
    def InfoGaint(Data):
        importances = Data.drop('Outcome', axis=1).apply(lambda x: x.corr(Data.Outcome))
        indices = np.argsort(importances)
        print(importances[indices])
        names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction' , 'Age']
        plt.title('Most important features (Information Gain)')
        plt.barh(range(len(indices)), importances[indices], color='#891eb3', align='center')
        plt.yticks(range(len(indices)), [names[i] for i in indices])
        plt.xlabel('Relative Importance')
        return plt.show()
    
#-------------------------------------------------------------------Main-------------------------------------------------------------------

# Create Object from our class
obj = Assignment3()

# Read Dataset
Data = obj.readDataSet('Assignment3_dataset.csv', 'Assignment3_dataset')
X = Data.iloc[:, :8]
Y = Data.iloc[:, [8]]

#-----------------------Q1-----------------------
x_train, x_test, y_train, y_test = obj.Split(X,Y,0.25,0)

#-----------------------1(a)-----------------------
KnnAcc = []
for i in range(1,11):
    KnnReport, Y_pred_Knn, Knn = obj.KNN(x_train,x_test,y_train,y_test,i)
    KnnAcc.append(obj.AccuracyTest(y_test,Y_pred_Knn))

# Plot the best Accuracies based number of neighbors
obj.Plot([1,2,3,4,5,6,7,8,9,10],KnnAcc,'k with Acc','#FF3300', 'o' , 100 , 'number of K' , 'Accuracies' , 'Best num Of K')

# Knn and Log
KnnReport, Y_pred_Knn, Knn = obj.KNN(x_train,x_test,y_train,y_test,6)
LogReport, Y_pred_Log, Log = obj.Logistic(x_train,x_test,y_train,y_test)

#-----------------------1(b)-----------------------
x_train_TSNE = obj.T_SNE(x_train)
x_test_TSNE = obj.T_SNE(x_test)

x_train_TSNE = pd.concat([pd.DataFrame(x_train_TSNE), pd.DataFrame(y_train)],axis=1 , ignore_index = True).astype(float)
x_test_TSNE = pd.concat([pd.DataFrame(x_test_TSNE), pd.DataFrame(y_test)],axis=1 , ignore_index = True).astype(float)

listOfclassesTrain = obj.GetListOfClasses(2, x_train_TSNE , pd.DataFrame(x_train_TSNE).columns[2])
listOfclassesTest = obj.GetListOfClasses(2, x_test_TSNE , pd.DataFrame(x_test_TSNE).columns[2])

# Plot Train and test dataset using T-SNE
obj.PlotDataPoints(2, listOfclassesTrain , 'Component 0','Component 1' ,'0','1' ,20 , 'T-SNE Train').show()
obj.PlotDataPoints(2, listOfclassesTest , 'Component 0','Component 1' ,'0','1', 20, 'T-SNE Test').show()

#-----------------------2(a)-----------------------
silhouette = []
for i in range(2,11):
    silhouette.append(obj.Kmeans(X, i, 0))
    
obj.Plot([2,3,4,5,6,7,8,9,10],silhouette,'scores with k','#009900', 'o' , 100 , 'Num of Clusters' , 'Silhouette Scores' , 'Silhouette with num of Clusters')

#-----------------------2(b)-----------------------
print("The best number of k (k-means) is = 2, because it's has the highest silhouette score which is = ",max(silhouette))

#-----------------------2(c)-----------------------
x_TSNE = obj.T_SNE(X)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

x_TSNE = pd.concat([pd.DataFrame(x_TSNE), pd.DataFrame(kmeans.labels_)],axis=1 , ignore_index = True).astype(float)
listOfclassesX = obj.GetListOfClasses(2, x_TSNE , pd.DataFrame(x_TSNE).columns[2])
obj.PlotDataPoints(2, listOfclassesX , 'Component 0','Component 1' ,'Cluster 0','Cluster 1' ,20 , 'K-means')
plt.show()

#-----------------------3(a)-----------------------
KnnAccPca = []
LogAccPca = []
for i in range(1,9):
    x_train_pca, x_test_pca = obj.Pca(x_train, x_test, i)
    KnnReportPca, Y_pred_Knn_pca, KnnPca = obj.KNN(x_train_pca,x_test_pca,y_train,y_test,6)
    LogReportPca, Y_pred_Log_Pca, LogPca = obj.Logistic(x_train_pca,x_test_pca,y_train,y_test)
    LogAccPca.append(obj.AccuracyTest(y_test,Y_pred_Log_Pca))
    KnnAccPca.append(obj.AccuracyTest(y_test,Y_pred_Knn_pca))

#-----------------------3(b)-----------------------
plt.plot([1,2,3,4,5,6,7,8], [68.75,68.75,68.75,68.75,68.75,68.75,68.75,68.75] , color = 'red')
obj.Plot([1,2,3,4,5,6,7,8],LogAccPca,'Logistic','#937DC2', 'o' , 100 , 'Num of PCA Components' , 'Accuracies' , 'Logistic with num of Components').show()

plt.plot([1,2,3,4,5,6,7,8], [67.1875,67.1875,67.1875,67.1875,67.1875,67.1875,67.1875,67.1875] , color = 'red')
obj.Plot([1,2,3,4,5,6,7,8],KnnAccPca,'KNN','#DF7861', 'o' , 100 , 'Num of PCA Components' , 'Accuracies' , 'Knn with num of Components').show()

obj.Bar([1,2,3,4,5,6,7,8], LogAccPca, 0.4, '#FF5B00' , 'Num of PCA Components', 'Accuracies' , 'Logistic and Knn With Pca', 'Logistic')
obj.Bar([1+0.3,2+0.3,3+0.3,4+0.3,5+0.3,6+0.3,7+0.3,8+0.3], KnnAccPca, 0.4, '#D4D925' , 'Num of PCA Components', 'Accuracies' , 'Logistic and Knn With Pca', 'Knn')
plt.show()

#-----------------------3(c)-----------------------
x_train_pca, x_test_pca = obj.Pca(x_train, x_test, 7) # 7 components was the highest accuracy for both knn and logistic regression
x_train_TSNE_pca = obj.T_SNE(x_train_pca)
x_test_TSNE_pca = obj.T_SNE(x_test_pca)

x_train_TSNE_pca = pd.concat([pd.DataFrame(x_train_TSNE_pca), pd.DataFrame(y_train)],axis=1 , ignore_index = True).astype(float)
x_test_TSNE_pca = pd.concat([pd.DataFrame(x_test_TSNE_pca), pd.DataFrame(y_test)],axis=1 , ignore_index = True).astype(float)

listOfclassesTrain_PCA = obj.GetListOfClasses(2, x_train_TSNE_pca , pd.DataFrame(x_train_TSNE_pca).columns[2])
listOfclassesTest_PCA = obj.GetListOfClasses(2, x_test_TSNE_pca , pd.DataFrame(x_test_TSNE_pca).columns[2])

# Plot PCA Train and test dataset using T-SNE
obj.PlotDataPoints(2, listOfclassesTrain_PCA , 'Component 0','Component 1' ,'0','1' ,20 , 'T-SNE 7 PCA Components Train').show()
obj.PlotDataPoints(2, listOfclassesTest_PCA , 'Component 0','Component 1' ,'0','1', 20, 'T-SNE 7 PCA Components Test').show()

#-----------------------4(a)-----------------------
#---------- Information Gain
FeaturePlt = obj.InfoGaint(Data)
FeatureImortance = mutual_info_classif(X,Y)
InfoGainAccKnn = []
InfoGainAccLog = []
for i in range(1,9):
    Xfeature = X.copy()[['Glucose', 'BMI', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction','Insulin','SkinThickness','BloodPressure']]
    Xfeature = Xfeature.iloc[:,0:i]
    x_trainInfo, x_testInfo, y_trainInfo, y_testInfo = obj.Split(Xfeature,Y,0.25,0)
    
    KnnReportinfo , y_pred_knn_info, modelknn_info = obj.KNN(x_trainInfo,x_testInfo,y_trainInfo,y_testInfo,6)
    InfoGainAccKnn.append(obj.AccuracyTest(y_test,y_pred_knn_info))
    
    LogReportinfo, Y_pred_Log_info, Loginfo = obj.Logistic(x_trainInfo,x_testInfo,y_trainInfo,y_testInfo)
    InfoGainAccLog.append(obj.AccuracyTest(y_test,Y_pred_Log_info))

# Plot
plt.plot([1,2,3,4,5,6,7,8], [70.833,70.833,70.833,70.833,70.833,70.833,70.833,70.833] , color = 'red')
obj.Plot([1,2,3,4,5,6,7,8],InfoGainAccKnn,'Knn with Best Num of Features','#408099', 'o' , 100 , 'Best Features (Information Gain)' , 'Accuracies' , 'Knn With Information Gain')

plt.plot([1,2,3,4,5,6,7,8], [76.041,76.041,76.041,76.041,76.041,76.041,76.041,76.041] , color = 'red')
obj.Plot([1,2,3,4,5,6,7,8],InfoGainAccLog,'Log with Best Num of Features','#946632', 'o' , 100 , 'Best Features (Information Gain)' , 'Accuracies' , 'Logistic With Information Gain')

obj.Bar([1,2,3,4,5,6,7,8], InfoGainAccLog, 0.4, '#a3de00' , 'Num of Best Features (Information Gain)', 'Accuracies' , 'Logistic and Knn With Information Gain', 'Logistic')
obj.Bar([1+0.3,2+0.3,3+0.3,4+0.3,5+0.3,6+0.3,7+0.3,8+0.3], InfoGainAccKnn, 0.4, '#00875c' , 'Num of Best Features (Information Gain)', 'Accuracies' , 'Logistic and Knn With Information Gain', 'Knn')
plt.show()

#-----------------------4(b)-----------------------
XForward = X.copy()

ForwardAccKnn = []
ForwardAccLog = []

# Logistic
for i in range(1,9):
    XForward = X.copy()
    sfs = SFS(LogisticRegression(),k_features=i,forward=True)
    XForward = sfs.fit_transform(XForward,Y)
    XForward = pd.DataFrame(XForward)
    x_trainForward, x_testForward, y_trainForward, y_testForward = obj.Split(XForward,Y,0.25,0)
    
    LogReportFor, Y_pred_Log_For, LogFor = obj.Logistic(x_trainForward,x_testForward,y_trainForward,y_testForward)
    ForwardAccLog.append(obj.AccuracyTest(y_test,Y_pred_Log_For))

# Knn
for i in range(1,9):
    XForward = X.copy()
    sfs = SFS(KNeighborsClassifier(n_neighbors=6),k_features=i,forward=True)
    XForward = sfs.fit_transform(XForward,Y)
    XForward = pd.DataFrame(XForward)
    x_trainForward, x_testForward, y_trainForward, y_testForward = obj.Split(XForward,Y,0.25,0)
    
    KnnReportFor , y_pred_knn_For, modelknn_For = obj.KNN(x_trainForward,x_testForward,y_trainForward,y_testForward,6)
    ForwardAccKnn.append(obj.AccuracyTest(y_test,y_pred_knn_For))
    
# Plot
plt.plot([1,2,3,4,5,6,7,8], [68.75,68.75,68.75,68.75,68.75,68.75,68.75,68.75] , color = 'red')
obj.Plot([1,2,3,4,5,6,7,8],ForwardAccKnn,'Knn with Best Num of Features','#408099', 'o' , 100 , 'Best Features (Forward Elimination)' , 'Accuracies' , 'Knn With Forward Elimination')

plt.plot([1,2,3,4,5,6,7,8], [76.5625,76.5625,76.5625,76.5625,76.5625,76.5625,76.5625,76.5625] , color = 'red')
obj.Plot([1,2,3,4,5,6,7,8],ForwardAccLog,'Log with Best Num of Features','#946632', 'o' , 100 , 'Best Features (Forward Elimination)' , 'Accuracies' , 'Logistic With Forward Elimination')

obj.Bar([1,2,3,4,5,6,7,8], ForwardAccLog, 0.4, '#7dd4ff' , 'Num of Best Features (Forward Elimination)', 'Accuracies' , 'Logistic and Knn With Forward Elimination', 'Logistic')
obj.Bar([1+0.3,2+0.3,3+0.3,4+0.3,5+0.3,6+0.3,7+0.3,8+0.3], ForwardAccKnn, 0.4, '#007dba' , 'Num of Best Features (Forward Elimination)', 'Accuracies' , 'Logistic and Knn With Forward Elimination', 'Knn')
plt.show()

#-----------------------4(c)-----------------------
XForward = X.copy()
sfs = SFS(LogisticRegression(),k_features=5,forward=True)

XForward = sfs.fit_transform(XForward,Y)
XForward = pd.DataFrame(XForward)
X_TSNE_For = obj.T_SNE(XForward)

x_trainForward, x_testForward, y_trainForward, y_testForward = obj.Split(XForward,Y,0.25,0)

x_train_TSNE_For = obj.T_SNE(x_trainForward)
x_test_TSNE_For = obj.T_SNE(x_testForward)

x_train_TSNE_For = pd.concat([pd.DataFrame(x_train_TSNE_For), pd.DataFrame(y_train)],axis=1 , ignore_index = True).astype(float)
x_test_TSNE_For = pd.concat([pd.DataFrame(x_test_TSNE_For), pd.DataFrame(y_test)],axis=1 , ignore_index = True).astype(float)

listOfclassesTrain_For = obj.GetListOfClasses(2, x_train_TSNE_For , pd.DataFrame(x_train_TSNE_For).columns[2])
listOfclassesTest_For = obj.GetListOfClasses(2, x_test_TSNE_For , pd.DataFrame(x_test_TSNE_For).columns[2])

obj.PlotDataPoints(2, listOfclassesTrain_For , 'Component 0','Component 1' ,'0','1' ,20 , 'T-SNE Forward elimination Best 5 Features Train').show()
obj.PlotDataPoints(2, listOfclassesTest_For , 'Component 0','Component 1' ,'0','1', 20, 'T-SNE Forward elimination Best 5 Features Test').show()


#-----------------------5(a)-----------------------
silhouetteForward = []
for i in range(2,11):
    silhouetteForward.append(obj.Kmeans(XForward, i, 0))
    
obj.Plot([2,3,4,5,6,7,8,9,10],silhouetteForward,'scores with k','#ff7300', 'o' , 100 , 'Num of Clusters' , 'Silhouette Scores' , 'Silhouette with num of Clusters (Forward Ekimination Best 5 Features)')

#-----------------------5(b)-----------------------
print("The best number of k (k-means with Forward Elimination) is = 3, because it's has the highest silhouette score which is = ",max(silhouetteForward))

kmeans = KMeans(n_clusters=3, random_state=0).fit(XForward)

X_TSNE_For = pd.concat([pd.DataFrame(X_TSNE_For), pd.DataFrame(kmeans.labels_)],axis=1 , ignore_index = True).astype(float)
listOfclassesX_For = obj.GetListOfClasses(3, X_TSNE_For , pd.DataFrame(X_TSNE_For).columns[2])

obj.PlotDataPoints3(3, listOfclassesX_For , 'Component 0','Component 1' ,'Cluster 0','Cluster 1', 'Cluster 2' ,20 , 'K-Means Forward Elimination Best 5 Features').show()

#-----------------------6(a)-----------------------
c=[]
t=[]
ncList=[]
accList=[]

#the next two def funtions are used to calculate the accuracy for unsupervised learning methods ONLY IF THE USED DATASET IS LABELD
def unsupervisedLabelMap(labels, y):
    labelDict = dict()
    for label in unique_labels(labels):
        tmpY = y[labels == label]
        unique, count = np.unique(tmpY, return_counts=True)
        trueLabel = unique[np.argmax(count)]
        labelDict[label] = trueLabel
    return labelDict


def usLabels2sLabels(labels, y):
    sLabels = np.empty(labels.shape, labels.dtype)
    labelDict = unsupervisedLabelMap(labels, y)
    for usl, tl in labelDict.items():
        sLabels[labels == usl] = tl
    return sLabels

rng=range(2,30)

for r in rng:
  model= SOM(m=r, n=1, dim=5)
  predClusters = model.fit_predict(XForward.to_numpy())
  predY = usLabels2sLabels(predClusters, Y.to_numpy())
  accuracy = accuracy_score(Y.to_numpy(), predY)
  score = silhouette_score(XForward.to_numpy(), predClusters, random_state=0)
  c.append(model.inertia_/len(XForward.to_numpy()))
  t.append(score)
  ncList.append(r)
  accList.append(accuracy)

#plot number of neurons/number of clusters vs silhoutte score
plt.plot(rng,t)
plt.xlabel("number of neurons")
plt.ylabel("Silhouette")
plt.show
       
#-----------------------6(b)-----------------------
print("The best number of neurons is = 2 because it's has the highest silhouette score which is = ",max(t))

#-----------------------6(c)-----------------------
som_shape = (1,2)
som = MiniSom(som_shape[0], som_shape[1], XForward.to_numpy().shape[1], sigma=.5, learning_rate=.5,
              neighborhood_function='gaussian', random_seed=10)

initial_weights = np.array(som.get_weights())
som.train_batch(XForward.to_numpy(), 1000, verbose=True)
final_weights = np.array(som.get_weights())

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in XForward.to_numpy()]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

for c in np.unique(cluster_index):
    plt.scatter(XForward.to_numpy()[cluster_index == c, 0],
                XForward.to_numpy()[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting initial centroids
plt.scatter(initial_weights[:, 0], initial_weights[:, 1], marker='x', 
                    s=80, linewidths=2, color='k', label='centroid')
plt.legend()
plt.title('SOM Initial Positions for BMUs/ Centroids')
plt.show()

for c in np.unique(cluster_index):
    plt.scatter(XForward.to_numpy()[cluster_index == c, 0],
                XForward.to_numpy()[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting final centroids
plt.scatter(final_weights[:, 0], final_weights[:, 1], marker='x', 
                s=80, linewidths=2, color='k', label='centroid')
plt.legend()
plt.title('SOM Final Positions for BMUs/ Centroids')
plt.show()

#-----------------------7(a)-----------------------
#find DBSCAN optimal eps and minpoints
epslist = np.array([0.3, 0.4, 0.5, 0.6,0.7])
minPoint = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15])

comb_array = np.array(np.meshgrid(epslist, minPoint)).T.reshape(-1, 2)

silhouetteDB = []
numberofcluster = []
epsls =[]
misls = []

for i in range(0,69):
    for j in range(0,69):
        model = DBSCAN(eps=comb_array[i][0], min_samples=comb_array[j][1])
        predLabels = model.fit_predict(XForward.to_numpy())
        if len(np.unique(predLabels)) == 1:
            continue
        else:
            epsls.append(comb_array[i][0])
            misls.append(comb_array[j][1])
            numberofcluster.append(len(np.unique(predLabels)))
            silhouetteDB.append(silhouette_score(XForward.to_numpy(), predLabels, metric='euclidean'))

EpsandMin = pd.concat([pd.DataFrame(silhouetteDB), pd.DataFrame(epsls), pd.DataFrame(misls), pd.DataFrame(numberofcluster)],axis=1 , ignore_index = True).astype(float)
EpsandMin.columns = ['silhouette_score', 'Eps', 'Min', 'NumofClusters']

# Based on the results from the EpsandMin
numofclus = [2,2,2,2,2,2,2,2,3,3]
Eps = [0.4,0.4,0.4,0.3,0.3,0.3,0.3,0.4,0.3,0.3]
Min = [14,11,10,9,7,10,13,2,6,5]

obj.Plot(numofclus,Eps,'Eps vs Num of Clusters','#408099', 'o' , 100 , 'Num of Clusters' , 'Epsilon' , 'Epsilon vs Num of Clusters')

#-----------------------7(b)-----------------------
obj.Plot(numofclus,Min,'MinPoints vs Num of Clusters','#408099', 'o' , 100 , 'Num of Clusters' , 'MinPoints' , 'MinPoints vs Num of Clusters')