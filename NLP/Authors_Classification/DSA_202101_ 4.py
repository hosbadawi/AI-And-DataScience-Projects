import nltk,re,pandas as pd,random,numpy as np
from nltk.corpus import stopwords 
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import  word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk import ngrams
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import sys
import subprocess

try : 
    import gensim
except:
    print("installing gensim ... ")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gensim'])
    
    
try : 
    import xgboost as xgb
except:
    print("installing xgb ... ")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost'])


# Impot the required libraries

class TextDataPartitioning:  # Create Class to generalize the program

    Partitioning_DataFrame1 = pd.DataFrame()  # Create two DataFrames for horizontal and vertical shapes
    Partitioning_DataFrame2 = pd.DataFrame()
    
    @staticmethod  # Use @staticmethod instead of using self as a function parameter
    def ReadBooks(Book_Name):  # This function will take Book_Name as parameter for example: austen-emma.txt
        return nltk.corpus.gutenberg.raw(Book_Name)  # Convert NLTK books to string datatype
    
    @staticmethod   
    def GetTitle(Book_Name):  # This function will take Book_Name as parameter for example: austen-emma.txt
        Book = TextDataPartitioning.ReadBooks(Book_Name)  # Call ReadBooks function
        TextDataPartitioning.Book_Title = re.findall('^\[(.*)\]', Book)  # Use re.findall to extract and assign the title of the book to list datatype variable
        TextDataPartitioning.Book_Title = TextDataPartitioning.Book_Title[0] # Convert list datatype to string datatype
        return TextDataPartitioning.Book_Title # Return book title as string datatype
    
    @staticmethod  # Use @staticmethod instead of using self as a function parameter
    def BookPreProcessing(Book_Name):   # This function will take Book_Name as parameter for example: austen-emma.txt
        Book = TextDataPartitioning.ReadBooks(Book_Name)  # Call ReadBooks function
        TextDataPartitioning.Book_Title = '[' + TextDataPartitioning.Book_Title + ']'  # Add [] to the book title to be able to remove it with the square brackets from the book
        Book = Book.replace(TextDataPartitioning.Book_Title,'')  # Remove book title from the book
        Book = re.sub('(CHAPTER(.*))|(Chapter(.*))', '', Book)  # Remove chapter title from the book
        Book = re.sub('(VOLUME(.*))|(Volume(.*))', '', Book)  # Remove Volume title from the book  
        Book = re.sub('^$\n', '', Book, flags = re.MULTILINE)  # Remove empty lines
        Book = re.sub('\. *(\W)','.\n\n', Book)  # Create a new line after each fullstop
        Book = re.sub('[^\w\s]','', Book)  # Remove punctuation marks from the book
        Book = re.sub(r'\b\w{1,2}\b', '',Book)
        Book = re.sub(r'(http|https|ftp)://[a-zA-Z0-9\\./]+', '', Book)
        Book = re.sub("[^a-zA-Z]", " ",Book)
        Book = Book.lower()  # Convert the book to small text
        Book = Book.split()  # Convert the book to list datatype
        StopWords = set(stopwords.words('english'))
        Book = [word for word in Book if not word in StopWords]  # Remove stopwords from this list
        return Book  # Return the book list datatype
    
    @staticmethod
    def BookPartitioning(Book_List):  # This function will take the book list to generate an equal partitions from it
       Book_List = [Book_List[i:i+100] for i in range(0, len(Book_List), 100)]  # Each partition will contain 100 sequential words
       random.shuffle(Book_List)  # Shuffle the partitions (random partitions from the book)
       Book_List = Book_List[0:200]  # The new book list will contain 200 random partitions from the book
       return Book_List  # Return the book list datatype   
    
    @staticmethod
    def CreateDataFrame1(Partitions_list, Book_Title):  # This function will return a 2 column dataframe the frist one (partitions) and the second one is (Books titles)
        Book_Names = [Book_Title] * len(Partitions_list)  # Create a list of len(partitions) and fill it with book title in each field 
        Temp_DF = pd.DataFrame({'Partitions': Partitions_list,'Authors': Book_Names})  # Create dataframe that will contain both (partitions column) and (books titles column)
        TextDataPartitioning.Partitioning_DataFrame2 = TextDataPartitioning.Partitioning_DataFrame2.append(Temp_DF, ignore_index=True)  # Append the temp dataframe to TextDataPartitioning.Partitioning_DataFrame2
        TextDataPartitioning.Partitioning_DataFrame2 = TextDataPartitioning.Partitioning_DataFrame2.sample(frac=1).reset_index(drop=True)  # Shuffle TextDataPartitioning.Partitioning_DataFrame2
        
        del Temp_DF  # Delete the temp dataframe
        return TextDataPartitioning.Partitioning_DataFrame2  # Return the dataframe
    
    @staticmethod
    def ConvertToString(DataFrame):
        for i in range(len(DataFrame)):
            DataFrame.iloc[i,0] = ' '.join([str(element) for element in DataFrame.iloc[i,0]])
        return DataFrame
    
    
    @staticmethod
    def Plot_Commaon_words(Paragaphs, Book_title,N_gram):
        
        parag=" ".join(Paragaphs)
        lst_tokens = nltk.tokenize.word_tokenize(parag)
        List = []
        for i in range(1,N_gram+1):
            dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, i))
            dtf_uni = pd.DataFrame(dic_words_freq.most_common(),  columns=["Word","Freq"])
            dtf_uni.sort_values(by="Freq" ,inplace=True ,ascending=False )
            dtf_uni.iloc[0:15 , :  ].set_index("Word").plot(kind="barh", title= f" {Book_title} {i}grams",legend=False).grid(axis='x')
            List.append(dtf_uni)
            plt.show()
            
        return List
    
    @staticmethod
    def LabelEncoder(Labels):
        le = preprocessing.LabelEncoder()
        Result = le.fit_transform(Labels)
        return Result
    
    @staticmethod
    def BoW_Encoder(Paragaphs):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(Paragaphs)
        return X
    
    @staticmethod
    def TFIDF_Encoder(Paragaphs):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(Paragaphs)
        return X
    
    @staticmethod
    def TFIDF_NGram_Encoder(Paragaphs , start_Ngram , End_Ngram):
        vectorizer = TfidfVectorizer(ngram_range=(start_Ngram,End_Ngram))
        X = vectorizer.fit_transform(Paragaphs)
        # print (list(vectorizer.vocabulary_.keys())[:10])
        return X
    
    @staticmethod
    def Toknaize_AllSet(Paragaphs):
        data = []
        for i in Paragaphs.values:
           temp = []
           for j in word_tokenize(i):
                temp.append(j.lower())
           data.append(temp)
        return data
    
    @staticmethod
    def PerFormLDA(X_train , y_train , X_test):
        lda_model = LDA(n_components = 4)
        X_train_lda = lda_model.fit_transform(X_train, y_train)
        X_test_lda = lda_model.transform(X_test)
        return  X_train_lda, X_test_lda
    
    @staticmethod
    def SplitDataSet(X,y):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25)
        return  X_train, X_test, y_train, y_test
    
    @staticmethod
    def Cross_Validation(Model , X , Y):
        skf = StratifiedKFold(n_splits=10, shuffle=False) 
        AccuriecseList = []
        ModelList = []
        
        # ClassifierXG_BOW_PCA = TextDataPartitioning.XG(X_trainBOW_PCA, y_train ,'multi:softmax',5, 0.3,0.1,5,8,10)
        # print(cross_val_score(ClassifierXG_BOW_PCA, X_trainBOW_PCA, y_train, cv=10, scoring='accuracy'))
        # ClassifierXG_BOW_PCA.score(X_trainBOW_PCA , y_train)
        
        
        for Tra_Idx, Tes_Idx in skf.split(X, Y): 
            X_train_fold, X_test_fold = X[Tra_Idx], X[Tes_Idx] 
            Y_train_fold, Y_test_fold = Y[Tra_Idx], Y[Tes_Idx] 
            
            Model.fit(X_train_fold, Y_train_fold)
            
            Y_predict= Model.predict(X_test_fold)
            AccuriecseList.append(accuracy_score( Y_test_fold , Y_predict))

            # AccuriecseList.append(Model.score(X_test_fold, Y_test_fold))
            ModelList.append(Model)
            
        return AccuriecseList , ModelList
    

    
    # SVM Classifier
    @staticmethod
    def SVM(X ,Y ,GeneralizationTerm):
        ClassifierSVM = SVC(kernel="rbf", C = GeneralizationTerm, probability=True )
        ClassifierSVM.fit(X,Y)
        return ClassifierSVM
    
    # DT Classifier
    @staticmethod
    def DecisionTree(X ,Y):
        ClassifierDT = DecisionTreeClassifier()
        ClassifierDT.fit(X,Y)
        return ClassifierDT
    
    # KNN Classifier
    @staticmethod
    def KNN(X ,Y ,K):
        ClassifierKNN = KNeighborsClassifier(n_neighbors=K)
        ClassifierKNN.fit(X, Y)
        return ClassifierKNN
    
    # Accuracy on Test
    @staticmethod
    def AccuracyTest(Y_Actual, Y_Pred):
        return accuracy_score(Y_Actual, Y_Pred) * 100
    
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
    
    # Function to detemine which features will be eliminated
    @staticmethod
    def FeatureSelection(X,Y, n):
        features = []
        Selector = ExtraTreesClassifier(n_estimators=n)
        Selector = Selector.fit(X, Y)
        features = Selector.feature_importances_
        return features
    
    @staticmethod
    def XG(X,Y,Objective, numOfClasses ,FeaturePrecentagePerTree , L_Rate , Max_depth , AlphaL1, TreeNum):
        xg_CLassification = xgb.XGBRegressor(objective = Objective,num_class = numOfClasses,  colsample_bytree = FeaturePrecentagePerTree, learning_rate = L_Rate,
                max_depth = Max_depth, alpha = AlphaL1, n_estimators = TreeNum)
        xg_CLassification.fit(X,Y)
        return xg_CLassification
    
    @staticmethod
    def PCA(XTrain,XTest,n):
        Pca = PCA(n_components = n)
        XTrain = Pca.fit_transform(XTrain)
        XTest = Pca.transform(XTest)
        return XTrain,XTest
    

    @staticmethod
    def plotImp(model, X , num = 20, fig_size = (40, 20)):
        feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})
        plt.figure(figsize=fig_size)
        sns.set(font_scale = 5)
        sns.barplot(x="Feature", y="Value", data=feature_imp.sort_values(by="Value", 
                    ascending=False)[0:num])
        plt.title('Features Importance')
        plt.tight_layout()
        plt.show()
        
    # Accuracy on Train
    @staticmethod
    def AccuracyTrain(Classifier,X_Train,Y_Train):
        return Classifier.score(X_Train, Y_Train) * 100
    
    # Accuracy on Train
    @staticmethod
    def BiasAndVariance(Y_pred, Y_test):
        Variance = np.var(Y_pred)
        
        SS_Errors = np.mean((np.mean(Y_pred) - Y_test)**2)

        
        Bias = SS_Errors - Variance
        
        return Bias, Variance
    
    @staticmethod
    def PLot_Accuracy_Comapre(AccuracyList , LabelsList , PLot_title ):
        
        sns.reset_defaults()
        fig = plt.figure(figsize = (12, 3))

        for  index ,elemnt in enumerate(AccuracyList):
            plt.title(PLot_title[index])


            plt.subplot(1, 3, index+1)
            plt.bar(LabelsList[index], elemnt, color ='maroon', width = 0.4)
            
        plt.show()
        
            
#---------------------------------------------------MAIN---------------------------------------------------    
sns.reset_defaults()

List_Of_All_Books = list(nltk.corpus.gutenberg.fileids())
print(List_Of_All_Books)
    
OBJ = TextDataPartitioning()

ListOfBooks = ['austen-emma.txt','edgeworth-parents.txt','melville-moby_dick.txt','chesterton-thursday.txt','carroll-alice.txt']
Authors = ['Jane Austen','Maria Edgeworth','Herman Melville','G. K. Chesterton','Lewis Carroll']

Book_Title = [None] * 5
Book = [None] * 5
Partitions = [None] * 5

for i in range(5):
    
    Book_Title[i] = TextDataPartitioning.GetTitle(ListOfBooks[i])
    
    Book[i] = TextDataPartitioning.BookPreProcessing(ListOfBooks[i])
    
    Partitions[i] = TextDataPartitioning.BookPartitioning(Book[i])
    
    DataFrame1 = TextDataPartitioning.CreateDataFrame1(Partitions[i], Authors[i])
    

print(DataFrame1.head(),'\n\n\n')
del Partitions


DataFrame1 = TextDataPartitioning.ConvertToString(DataFrame1)

DataFrame1Copy2 = DataFrame1.copy()



TextDataPartitioning.Plot_Commaon_words(DataFrame1Copy2[ DataFrame1Copy2["Authors" ]== "Jane Austen"]["Partitions"] , "Jane Austen" , 1 )
TextDataPartitioning.Plot_Commaon_words(DataFrame1Copy2[ DataFrame1Copy2["Authors" ]== "Maria Edgeworth"]["Partitions"] , DataFrame1Copy2["Authors"][1] , 1 )
TextDataPartitioning.Plot_Commaon_words(DataFrame1Copy2[ DataFrame1Copy2["Authors" ]== "Herman Melville"]["Partitions"] , DataFrame1Copy2["Authors"][2] , 1 )
TextDataPartitioning.Plot_Commaon_words(DataFrame1Copy2[ DataFrame1Copy2["Authors" ]== "G. K. Chesterton"]["Partitions"] , DataFrame1Copy2["Authors"][3] , 1 )
TextDataPartitioning.Plot_Commaon_words(DataFrame1Copy2[ DataFrame1Copy2["Authors" ]== "Lewis Carroll"]["Partitions"] , DataFrame1Copy2["Authors"][4] , 1 )

DataFrame1.Authors = TextDataPartitioning.LabelEncoder(DataFrame1.Authors)

X = DataFrame1.Partitions
Y = DataFrame1.Authors

# BOW
X_BOW = TextDataPartitioning.BoW_Encoder(X).toarray()

# TF-IDF
X_TFIDF = TextDataPartitioning.TFIDF_Encoder(X).toarray()

# N-Gram
X_NGRAM = TextDataPartitioning.TFIDF_NGram_Encoder(X, 1,2).toarray()

X_trainBOW, X_testBOW, y_train, y_test = TextDataPartitioning.SplitDataSet(X_BOW,Y)
X_trainTFIDF, X_testTFIDF, y_train, y_test = TextDataPartitioning.SplitDataSet(X_TFIDF,Y)
X_trainNGRAM, X_testNGRAM, y_train, y_test = TextDataPartitioning.SplitDataSet(X_NGRAM,Y)

y_train, y_test = y_train.reset_index(drop=True), y_test.reset_index(drop=True)

# LDA
X_trainBOW_LDA , X_testBOW_LDA = TextDataPartitioning.PerFormLDA(X_trainBOW, y_train, X_testBOW)
X_trainTFIDF_LDA , X_testTFIDF_LDA = TextDataPartitioning.PerFormLDA(X_trainTFIDF, y_train, X_testTFIDF)
X_trainNGRAM_LDA , X_testNGRAM_LDA = TextDataPartitioning.PerFormLDA(X_trainNGRAM, y_train, X_testNGRAM)

# PCA
X_trainBOW_PCA , X_testBOW_PCA = TextDataPartitioning.PCA(X_trainBOW, X_testBOW,650)
X_trainTFIDF_PCA , X_testTFIDF_PCA = TextDataPartitioning.PCA(X_trainTFIDF,X_testTFIDF,650)
X_trainNGRAM_PCA , X_testNGRAM_PCA = TextDataPartitioning.PCA(X_trainNGRAM,X_testNGRAM,650)

# Most_ImpotentFeatures_BOW_Train , Most_ImpotentFeatures_BOW_Test = TextDataPartitioning.PCA(X_trainBOW, X_testBOW,20)

# Word Embedding
# X_w2v = TextDataPartitioning.Word2Vec(X)

# Append in the lists to find the champion model
MaxMumAcc = []
MaxMumAccModels = []

MaxBais = []
MaxVariance = []

Model_Names = []

Preicison = []

Recall = []

F1_Score = []


#---------------------------------------SVM---------------------------------------


ClassifierSVM_BOW = TextDataPartitioning.SVM(X_trainBOW_LDA, y_train, 100)

ClassifierSVM_TFIDF = TextDataPartitioning.SVM(X_trainTFIDF_LDA, y_train, 100)

ClassifierSVM_NGRAM = TextDataPartitioning.SVM(X_trainNGRAM_LDA, y_train, 100)


Y_PredSVM_BOW = TextDataPartitioning.Pred(ClassifierSVM_BOW,X_testBOW_LDA)
Y_PredSVM_TFIDF = TextDataPartitioning.Pred(ClassifierSVM_TFIDF,X_testTFIDF_LDA)
Y_PredSVM_NGRAM = TextDataPartitioning.Pred(ClassifierSVM_NGRAM,X_testNGRAM_LDA)

AccSVM_OnTest_BOW = TextDataPartitioning.AccuracyTest(y_test, Y_PredSVM_BOW)
AccSVM_OnTest_TFIDF  = TextDataPartitioning.AccuracyTest(y_test, Y_PredSVM_TFIDF)
AccSVM_OnTest_NGRAM = TextDataPartitioning.AccuracyTest(y_test, Y_PredSVM_NGRAM)

AccSVM_OnTrain_BOW = TextDataPartitioning.AccuracyTrain(ClassifierSVM_BOW, X_trainBOW_LDA, y_train)
AccSVM_OnTrain_TFIDF = TextDataPartitioning.AccuracyTrain(ClassifierSVM_TFIDF, X_trainTFIDF_LDA, y_train)
AccSVM_OnTrain_NGRAM = TextDataPartitioning.AccuracyTrain(ClassifierSVM_NGRAM, X_trainNGRAM_LDA, y_train)

# Plot Confusion Matrix
CF_MatrixSVM_BOW = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredSVM_BOW)
CF_MatrixSVM_TFIDF = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredSVM_TFIDF)
CF_MatrixSVM_NGRAM = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredSVM_NGRAM)

TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixSVM_BOW,'BOW-LDA-SVM')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixSVM_TFIDF,'TF-IDF-LDA-SVM')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixSVM_NGRAM,'N-Gram-LDA-SVM')

# Cross-Validation
SVM_BOW_Cross ,SVM_BOW_Models = TextDataPartitioning.Cross_Validation(ClassifierSVM_BOW,X_trainBOW_LDA, y_train)
SVM_TFIDF_Cross , SVM_TFIDF_Models = TextDataPartitioning.Cross_Validation(ClassifierSVM_TFIDF,X_trainTFIDF_LDA, y_train)
SVM_NGRAM_Cross ,SVM_NGRAM__Models = TextDataPartitioning.Cross_Validation(ClassifierSVM_NGRAM,X_trainNGRAM_LDA, y_train)

print('SVM_BOW_Accurices : \n', SVM_BOW_Cross , '\n\n','SVM_TFIDF_Accurices : \n', SVM_TFIDF_Cross ,'\n\n','SVM_NGRAM_Accurices : \n',SVM_NGRAM_Cross,'\n\n\n')




Model_Names1= ["SVM_BOW", "SVM_TFIDF" , "SVM_NGRAM"   ]
# # PLot Compere etween K_Foldes For Each Model
Names = [0,1,2,3,4,5,6,7,8,9]
SVM_Scores = [SVM_BOW_Cross,SVM_TFIDF_Cross,SVM_NGRAM_Cross]
TextDataPartitioning.PLot_Accuracy_Comapre(SVM_Scores,[Names,Names,Names],Model_Names1)

# GetMax Accuracy
indexs =  np.unravel_index(np.array(SVM_Scores).argmax(), np.array(SVM_Scores).shape)

MaxAcc_Model = np.array([SVM_BOW_Models,SVM_TFIDF_Models,SVM_NGRAM__Models ])[indexs[0],indexs[1]]

Y_predicted = [Y_PredSVM_BOW , Y_PredSVM_TFIDF ,Y_PredSVM_NGRAM ]

BiasSVM_BOW_LDA , VarianceSVM_BOW_LDA = TextDataPartitioning.BiasAndVariance(Y_predicted[indexs[0]] , y_test)

P_R_F =  precision_recall_fscore_support(y_test, Y_predicted[indexs[0]], average='macro')


print("-------------------")

print(f"The Maxmum accuracy : {np.array(SVM_Scores)[indexs[0],indexs[1]]}")
print(f"With Bias : {BiasSVM_BOW_LDA}")
print(f"With Varince : {VarianceSVM_BOW_LDA}")

print("-------------------")

print(f"precision : {round(P_R_F[0],3)}")
print(f"recall : {round(P_R_F[1],3)}")
print(f"F1 Score : {round(P_R_F[2],3)}")

print("-------------------")


Preicison.append(P_R_F[0])

Recall.append(P_R_F[1])

F1_Score.append(P_R_F[2])

print('\n\n')


MaxMumAccModels.append(MaxAcc_Model)
MaxMumAcc.append(np.array(SVM_Scores)[indexs[0],indexs[1]])
MaxBais.append(BiasSVM_BOW_LDA)
MaxVariance.append(VarianceSVM_BOW_LDA)
Model_Names.append(Model_Names1[indexs[0]])



# #---------------------------------------DecisionTree---------------------------------------
ClassifierDT_BOW = TextDataPartitioning.DecisionTree(X_trainBOW, y_train)

ClassifierDT_TFIDF = TextDataPartitioning.DecisionTree(X_trainTFIDF, y_train)

ClassifierDT_NGRAM = TextDataPartitioning.DecisionTree(X_trainNGRAM, y_train)



Y_PredDT_BOW = TextDataPartitioning.Pred(ClassifierDT_BOW,X_testBOW)

Y_PredDT_TFIDF = TextDataPartitioning.Pred(ClassifierDT_TFIDF,X_testTFIDF)

Y_PredDT_NGRAM = TextDataPartitioning.Pred(ClassifierDT_NGRAM,X_testNGRAM)


AccDT_OnTest_BOW = TextDataPartitioning.AccuracyTest(y_test, Y_PredDT_BOW)

AccDT_OnTest_TFIDF  = TextDataPartitioning.AccuracyTest(y_test, Y_PredDT_TFIDF)

AccDT_OnTest_NGRAM = TextDataPartitioning.AccuracyTest(y_test, Y_PredDT_NGRAM)

AccDT_OnTrain_BOW = TextDataPartitioning.AccuracyTrain(ClassifierDT_BOW, X_trainBOW, y_train)

AccDT_OnTrain_TFIDF = TextDataPartitioning.AccuracyTrain(ClassifierDT_TFIDF, X_trainTFIDF, y_train)

AccDT_OnTrain_NGRAM = TextDataPartitioning.AccuracyTrain(ClassifierDT_NGRAM, X_trainNGRAM, y_train)

# Plot Confusion Matrix
CF_MatrixDT_BOW = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredDT_BOW)
CF_MatrixDT_TFIDF = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredDT_TFIDF)
CF_MatrixDT_NGRAM = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredDT_NGRAM)


TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixDT_BOW,'BOW-DT')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixDT_TFIDF,'TF-IDF-DT')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixDT_NGRAM,'N-Gram-DT')

# Cross-Validation
DT_BOW_Cross,DT_BOW_Cross_Models = TextDataPartitioning.Cross_Validation(ClassifierDT_BOW,X_trainBOW, y_train)

DT_TFIDF_Cross,DT_BOW_Cross_Models = TextDataPartitioning.Cross_Validation(ClassifierDT_TFIDF,X_trainTFIDF, y_train)

DT_NGRAM_Cross,DT_BOW_Cross_Models = TextDataPartitioning.Cross_Validation(ClassifierDT_NGRAM,X_trainNGRAM, y_train)

print('DT_BOW_Accurices : \n', DT_BOW_Cross , '\n\n','DT_TFIDF_Accurices : \n', DT_TFIDF_Cross ,'\n\n','DT_NGRAM_Accurices : \n', DT_NGRAM_Cross,'\n\n\n')

Model_Names1= ["DT_BOW", "DT_TFIDF" , "DT_NGRAM"  ]


DT_Scores = [DT_BOW_Cross,DT_TFIDF_Cross,DT_NGRAM_Cross]



# X_Cham_Test = [X_testBOW , X_testTFIDF , X_testNGRAM]

# MaxAcc_Model.predict( X_Cham_Test [ idnex [0]  ]  )

TextDataPartitioning.PLot_Accuracy_Comapre(DT_Scores,[Names,Names,Names],Model_Names1)

# GetMax Accuracy
indexs =  np.unravel_index(np.array(DT_Scores).argmax(), np.array(DT_Scores).shape)

MaxAcc_Model = np.array([DT_BOW_Cross_Models,DT_BOW_Cross_Models,DT_BOW_Cross_Models ])[indexs[0],indexs[1]]




Y_predicted = [Y_PredDT_BOW , Y_PredDT_TFIDF ,Y_PredDT_NGRAM ]

BiasSVM_BOW_LDA , VarianceSVM_BOW_LDA = TextDataPartitioning.BiasAndVariance(Y_predicted[indexs[0]] , y_test)

P_R_F =  precision_recall_fscore_support(y_test, Y_predicted[indexs[0]], average='macro')



print("-------------------")
print(f"The Maxmum accuracy : {np.array(DT_Scores)[indexs[0],indexs[1]]}")
print(f"With Bias : {BiasSVM_BOW_LDA}")
print(f"With Varince : {VarianceSVM_BOW_LDA}")

print("-------------------")

print(f"precision : {round(P_R_F[0],3)}")
print(f"recall : {round(P_R_F[1],3)}")
print(f"F1 Score : {round(P_R_F[2],3)}")

print("-------------------")

print('\n\n')

Preicison.append(P_R_F[0])

Recall.append(P_R_F[1])

F1_Score.append(P_R_F[2])


MaxMumAccModels.append(MaxAcc_Model)
MaxMumAcc.append(np.array(DT_Scores)[indexs[0],indexs[1]])
MaxBais.append(BiasSVM_BOW_LDA)
MaxVariance.append(VarianceSVM_BOW_LDA)
Model_Names.append(Model_Names1[indexs[0]])

Preicison.append(P_R_F[0])

Recall.append(P_R_F[1])

F1_Score.append(P_R_F[2])





#---------------------------------------KNN---------------------------------------

ClassifierKNN_BOW = TextDataPartitioning.KNN(X_trainBOW, y_train,5)
ClassifierKNN_TFIDF = TextDataPartitioning.KNN(X_trainTFIDF, y_train,5)
ClassifierKNN_NGRAM = TextDataPartitioning.KNN(X_trainNGRAM, y_train,5)

Y_PredKNN_BOW = TextDataPartitioning.Pred(ClassifierKNN_BOW,X_testBOW)
Y_PredKNN_TFIDF = TextDataPartitioning.Pred(ClassifierKNN_TFIDF,X_testTFIDF)
Y_PredKNN_NGRAM = TextDataPartitioning.Pred(ClassifierKNN_NGRAM,X_testNGRAM)

AccKNN_OnTest_BOW = TextDataPartitioning.AccuracyTest(y_test, Y_PredKNN_BOW)
AccKNN_OnTest_TFIDF  = TextDataPartitioning.AccuracyTest(y_test, Y_PredKNN_TFIDF)
AccKNN_OnTest_NGRAM = TextDataPartitioning.AccuracyTest(y_test, Y_PredKNN_NGRAM)

AccKNN_OnTrain_BOW = TextDataPartitioning.AccuracyTrain(ClassifierKNN_BOW, X_trainBOW, y_train)
AccKNN_OnTrain_TFIDF = TextDataPartitioning.AccuracyTrain(ClassifierKNN_TFIDF, X_trainTFIDF, y_train)
AccKNN_OnTrain_NGRAM = TextDataPartitioning.AccuracyTrain(ClassifierKNN_NGRAM, X_trainNGRAM, y_train)

# Plot Confusion Matrix
CF_MatrixKNN_BOW = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredKNN_BOW)
CF_MatrixKNN_TFIDF = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredKNN_TFIDF)
CF_MatrixKNN_NGRAM = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredKNN_NGRAM)

TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixKNN_BOW,'BOW-KNN')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixKNN_TFIDF,'TF-IDF-KNN')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixKNN_NGRAM,'N-Gram-KNN')

# Cross-Validation
KNN_BOW_Cross ,KNN_BOW_Cross_Model = TextDataPartitioning.Cross_Validation(ClassifierKNN_BOW,X_trainBOW, y_train)
KNN_TFIDF_Cross ,KNN_TFIDF_Cross_Model = TextDataPartitioning.Cross_Validation(ClassifierKNN_TFIDF,X_trainTFIDF, y_train)
KNN_NGRAM_Cross ,KNN_NGRAM_Cross_Model = TextDataPartitioning.Cross_Validation(ClassifierKNN_NGRAM,X_trainNGRAM, y_train)

print('KNN_BOW_Accurices : \n', KNN_BOW_Cross , '\n\n','KNN_TFIDF_Accurices : \n', KNN_TFIDF_Cross ,'\n\n','KNN_NGRAM_Accurices : \n', KNN_NGRAM_Cross , '\n\n\n')

Model_Names1= ["KNN_BOW", "KNN_TFIDF" , "KNN_NGRAM"  ]

KNN_Scores = [KNN_BOW_Cross,KNN_TFIDF_Cross,KNN_NGRAM_Cross]
TextDataPartitioning.PLot_Accuracy_Comapre(KNN_Scores,[Names,Names,Names],Model_Names1)

# GetMax Accuracy
indexs =  np.unravel_index(np.array(KNN_Scores).argmax(), np.array(KNN_Scores).shape)

MaxAcc_Model = np.array([KNN_BOW_Cross_Model,KNN_TFIDF_Cross_Model,KNN_NGRAM_Cross_Model ])[indexs[0],indexs[1]]

Y_predicted = [Y_PredKNN_BOW , Y_PredKNN_TFIDF ,Y_PredKNN_NGRAM ]

BiasSVM_BOW_LDA , VarianceSVM_BOW_LDA = TextDataPartitioning.BiasAndVariance(Y_predicted[indexs[0]] , y_test)


P_R_F =  precision_recall_fscore_support(y_test, Y_predicted[indexs[0]], average='macro')


print("-------------------")
print(f"The Maxmum accuracy : {np.array(KNN_Scores)[indexs[0],indexs[1]]}")
print(f"With Bias : {BiasSVM_BOW_LDA}")
print(f"With Varince : {VarianceSVM_BOW_LDA}")
print("-------------------")

print(f"precision : {round(P_R_F[0],3)}")
print(f"recall : {round(P_R_F[1],3)}")
print(f"F1 Score : {round(P_R_F[2],3)}")

print("-------------------")

print('\n\n')


MaxMumAccModels.append(MaxAcc_Model)
MaxMumAcc.append(np.array(KNN_Scores)[indexs[0],indexs[1]])
MaxBais.append(BiasSVM_BOW_LDA)
MaxVariance.append(VarianceSVM_BOW_LDA)


Model_Names.append(Model_Names1[indexs[0]])

Preicison.append(P_R_F[0])

Recall.append(P_R_F[1])

F1_Score.append(P_R_F[2])

# #---------------------------------------XG---------------------------------------

ClassifierXG_BOW = TextDataPartitioning.XG(X_trainBOW, y_train ,'multi:softmax',5, 0.3,0.1,5,8,10)
ClassifierXG_TFIDF = TextDataPartitioning.XG(X_trainTFIDF, y_train,'multi:softmax',5, 0.3,0.1,5,8,10)
ClassifierXG_NGRAM = TextDataPartitioning.XG(X_trainNGRAM, y_train ,'multi:softmax',5,0.3,0.1,5,8,10)

Y_PredXG_BOW = TextDataPartitioning.Pred(ClassifierXG_BOW,X_testBOW)
Y_PredXG_TFIDF = TextDataPartitioning.Pred(ClassifierXG_TFIDF,X_testTFIDF)
Y_PredXG_NGRAM = TextDataPartitioning.Pred(ClassifierXG_NGRAM,X_testNGRAM)

AccXG_OnTest_BOW = TextDataPartitioning.AccuracyTest(y_test, Y_PredXG_BOW)
AccXG_OnTest_TFIDF  = TextDataPartitioning.AccuracyTest(y_test, Y_PredXG_TFIDF)
AccXG_OnTest_NGRAM = TextDataPartitioning.AccuracyTest(y_test, Y_PredXG_NGRAM)

AccXG_OnTrain_BOW = TextDataPartitioning.AccuracyTrain(ClassifierXG_BOW, X_trainBOW, y_train)
AccXG_OnTrain_TFIDF = TextDataPartitioning.AccuracyTrain(ClassifierXG_TFIDF, X_trainTFIDF, y_train)
AccXG_OnTrain_NGRAM = TextDataPartitioning.AccuracyTrain(ClassifierXG_NGRAM, X_trainNGRAM, y_train)

# Plot Confusion Matrix
CF_MatrixXG_BOW = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredXG_BOW)
CF_MatrixXG_TFIDF = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredXG_TFIDF)
CF_MatrixXG_NGRAM = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredXG_NGRAM)

TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixXG_BOW,'BOW-XG')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixXG_TFIDF,'TF-IDF-XG')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixXG_NGRAM,'N-Gram-XG')

# Cross-Validation
XG_BOW_Cross ,XG_BOW_Cross_Model  = TextDataPartitioning.Cross_Validation(ClassifierXG_BOW,X_trainBOW, y_train)
XG_TFIDF_Cross , XG_BOW_Cross_Model = TextDataPartitioning.Cross_Validation(ClassifierXG_TFIDF,X_trainTFIDF, y_train)
XG_NGRAM_Cross  , XG_BOW_Cross_Model= TextDataPartitioning.Cross_Validation(ClassifierXG_NGRAM,X_trainNGRAM, y_train)

print('XG_BOW_Accurices : \n', XG_BOW_Cross , '\n\n','XG_TFIDF_Accurices : \n', XG_TFIDF_Cross ,'\n\n','XG_NGRAM_Accurices : \n', XG_NGRAM_Cross,'\n\n\n')
Model_Names1= ["XG_BOW", "XG_TFIDF" , "XG_NGRAM"  ]


XG_Scores = [XG_BOW_Cross,XG_TFIDF_Cross,XG_NGRAM_Cross]
TextDataPartitioning.PLot_Accuracy_Comapre(XG_Scores,[Names,Names,Names],Model_Names1)


# GetMax Accuracy
indexs =  np.unravel_index(np.array(XG_Scores).argmax(), np.array(XG_Scores).shape)

MaxAcc_Model = np.array([XG_BOW_Cross_Model,XG_BOW_Cross_Model,XG_BOW_Cross_Model ])[indexs[0],indexs[1]]

Y_predicted = [Y_PredXG_BOW , Y_PredXG_TFIDF ,Y_PredXG_NGRAM ]

BiasSVM_BOW_LDA , VarianceSVM_BOW_LDA = TextDataPartitioning.BiasAndVariance(Y_predicted[indexs[0]] , y_test)



P_R_F =  precision_recall_fscore_support(y_test, Y_predicted[indexs[0]], average='macro')


print("-------------------")
print(f"The Maxmum accuracy : {np.array(XG_Scores)[indexs[0],indexs[1]]}")
print(f"With Bias : {BiasSVM_BOW_LDA}")
print(f"With Varince : {VarianceSVM_BOW_LDA}")
print("-------------------")

print(f"precision : {round(P_R_F[0],3)}")
print(f"recall : {round(P_R_F[1],3)}")
print(f"F1 Score : {round(P_R_F[2],3)}")

print("-------------------")

print('\n\n')


MaxMumAccModels.append(MaxAcc_Model)
MaxMumAcc.append(np.array(XG_Scores)[indexs[0],indexs[1]])
MaxBais.append(BiasSVM_BOW_LDA)
MaxVariance.append(VarianceSVM_BOW_LDA)


Model_Names.append(Model_Names1[indexs[0]])

Preicison.append(P_R_F[0])

Recall.append(P_R_F[1])

F1_Score.append(P_R_F[2])



#---------------------------------------XG_PCA---------------------------------------
ClassifierXG_BOW_PCA = TextDataPartitioning.XG(X_trainBOW_PCA, y_train ,'multi:softmax',5, 0.3,0.1,5,8,10)
ClassifierXG_TFIDF_PCA = TextDataPartitioning.XG(X_trainTFIDF_PCA, y_train,'multi:softmax',5, 0.3,0.1,5,8,10)
ClassifierXG_NGRAM_PCA = TextDataPartitioning.XG(X_trainNGRAM_PCA, y_train ,'multi:softmax',5,0.3,0.1,5,8,10)

Y_PredXG_BOW_PCA = TextDataPartitioning.Pred(ClassifierXG_BOW_PCA,X_testBOW_PCA)
Y_PredXG_TFIDF_PCA = TextDataPartitioning.Pred(ClassifierXG_TFIDF_PCA,X_testTFIDF_PCA)
Y_PredXG_NGRAM_PCA = TextDataPartitioning.Pred(ClassifierXG_NGRAM_PCA,X_testNGRAM_PCA)

AccXG_OnTest_BOW_PCA = TextDataPartitioning.AccuracyTest(y_test, Y_PredXG_BOW_PCA)
AccXG_OnTest_TFIDF_PCA  = TextDataPartitioning.AccuracyTest(y_test, Y_PredXG_TFIDF_PCA)
AccXG_OnTest_NGRAM_PCA = TextDataPartitioning.AccuracyTest(y_test, Y_PredXG_NGRAM_PCA)

AccXG_OnTrain_BOW_PCA = TextDataPartitioning.AccuracyTrain(ClassifierXG_BOW_PCA, X_trainBOW_PCA, y_train)
AccXG_OnTrain_TFIDF_PCA = TextDataPartitioning.AccuracyTrain(ClassifierXG_TFIDF_PCA, X_trainTFIDF_PCA, y_train)
AccXG_OnTrain_NGRAM_PCA = TextDataPartitioning.AccuracyTrain(ClassifierXG_NGRAM_PCA, X_trainNGRAM_PCA, y_train)

# Plot Confusion Matrix
CF_MatrixXG_BOW_PCA = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredXG_BOW_PCA)
CF_MatrixXG_TFIDF_PCA = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredXG_TFIDF_PCA)
CF_MatrixXG_NGRAM_PCA = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredXG_NGRAM_PCA)

TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixXG_BOW_PCA,'BOW-XG-PCA')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixXG_TFIDF_PCA,'TF-IDF-XG-PCA')
TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixXG_NGRAM_PCA,'N-Gram-XG-PCA')

# Cross-Validation
XG_BOW_Cross_PCA , XG_BOW_Cross_PCA_Model  = TextDataPartitioning.Cross_Validation(ClassifierXG_BOW_PCA,X_trainBOW_PCA, y_train)
XG_TFIDF_Cross_PCA ,XG_TFIDF_Cross_PCA_Model = TextDataPartitioning.Cross_Validation(ClassifierXG_TFIDF_PCA,X_trainTFIDF_PCA, y_train)
XG_NGRAM_Cross_PCA , XG_NGRAM_Cross_PCA_Model = TextDataPartitioning.Cross_Validation(ClassifierXG_NGRAM_PCA,X_trainNGRAM_PCA, y_train)

print('XG_BOW__PCA_Accurices : \n', XG_BOW_Cross_PCA , '\n\n','XG_TFIDF_PCA_Accurices : \n', XG_TFIDF_Cross_PCA ,'\n\n','XG_NGRAM_PCA_Accurices : \n', XG_NGRAM_Cross_PCA)

Model_Names1= ["XG_BOW_PCA", "XG_TFIDF_PCA" , "XG_NGRAM_PCA"  ]


XG_PCA_Scores = [XG_BOW_Cross_PCA,XG_TFIDF_Cross_PCA,XG_NGRAM_Cross_PCA]
TextDataPartitioning.PLot_Accuracy_Comapre(XG_PCA_Scores,[Names,Names,Names],Model_Names1)

# GetMax Accuracy
indexs =  np.unravel_index(np.array(XG_PCA_Scores).argmax(), np.array(XG_PCA_Scores).shape)

MaxAcc_Model = np.array([XG_BOW_Cross_PCA_Model,XG_TFIDF_Cross_PCA_Model,XG_NGRAM_Cross_PCA_Model ])[indexs[0],indexs[1]]

Y_predicted = [Y_PredXG_BOW_PCA , Y_PredXG_TFIDF_PCA ,Y_PredXG_NGRAM_PCA ]

BiasSVM_BOW_LDA , VarianceSVM_BOW_LDA = TextDataPartitioning.BiasAndVariance(Y_predicted[indexs[0]] , y_test)



P_R_F =  precision_recall_fscore_support(y_test, Y_predicted[indexs[0]], average='macro')

print("-------------------")
print(f"The Maxmum accuracy : {np.array(XG_PCA_Scores)[indexs[0],indexs[1]]}")
print(f"With Bias : {BiasSVM_BOW_LDA}")
print(f"With Varince : {VarianceSVM_BOW_LDA}")
print("-------------------")

print(f"precision : {round(P_R_F[0],3)}")
print(f"recall : {round(P_R_F[1],3)}")
print(f"F1 Score : {round(P_R_F[2],3)}")

print("-------------------")
print('\n\n')


MaxMumAccModels.append(MaxAcc_Model)
MaxMumAcc.append(np.array(XG_PCA_Scores)[indexs[0],indexs[1]])
MaxBais.append(BiasSVM_BOW_LDA)
MaxVariance.append(VarianceSVM_BOW_LDA)



Model_Names.append(Model_Names1[indexs[0]])

Preicison.append(P_R_F[0])

Recall.append(P_R_F[1])

F1_Score.append(P_R_F[2])

# #---------------------------------------------Analysis of Bias and Variability-----------------------------------------------------------------

# #--------------------------------------------- Caculate Maximum Acuuracy -----------------------------------------------------------------




BestModelIndex = np.argmax(MaxMumAcc)



print("---------------------------------------------------------------")

print("-------------------")
print(f"Champion Model is {Model_Names[BestModelIndex]}")
print("-------------------")

print(f"Champion Model Accuracy : {MaxMumAcc[BestModelIndex]}")
print(f"Champion Model Baise : {MaxBais[BestModelIndex]}")
print(f"Champion Model Varince : {MaxVariance[BestModelIndex]}")
print("-------------------")
print('\n\n')



Test_Data , Train_Data =[],[]

if "PCA"  in  Model_Names[BestModelIndex]:
    
    if "TFIDF"  in  Model_Names[BestModelIndex]:
            Test_Data = X_testTFIDF_PCA
            Train_Data = X_trainTFIDF_PCA 
    elif "BOW"  in  Model_Names[BestModelIndex]:
          Test_Data = X_testBOW_PCA
          Train_Data = X_trainBOW_PCA
    elif "NGRAM"  in  Model_Names[BestModelIndex]:
          Test_Data = X_testNGRAM_PCA
          Train_Data = X_trainBOW_PCA
    

elif "TFIDF"  in  Model_Names[BestModelIndex]:
        Test_Data = X_testTFIDF
        Train_Data = X_trainTFIDF
elif "BOW"  in  Model_Names[BestModelIndex]:
      Test_Data = X_testBOW
      Train_Data = X_trainBOW
elif "NGRAM"  in  Model_Names[BestModelIndex]:
      Test_Data = X_testNGRAM
      Train_Data = X_trainNGRAM
# Test Model And PLot Confucsion Matrix


Y_PredKNN_TFIDF = TextDataPartitioning.Pred( MaxMumAccModels[BestModelIndex] , Test_Data)

AccKNN_OnTest_TFIDF  = TextDataPartitioning.AccuracyTest(y_test, Y_PredKNN_TFIDF)



AccKNN_OnTrain_TFIDF = TextDataPartitioning.AccuracyTrain( MaxMumAccModels[BestModelIndex], Train_Data, y_train)

# # Plot Confusion Matrix
CF_MatrixKNN_TFIDF = TextDataPartitioning.ConfusionMatrix(y_test, Y_PredKNN_TFIDF)

TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixKNN_TFIDF,'Champion Model')





sns.reset_defaults()
fig = plt.figure(figsize = (10, 5))

plt.title("champion Model Train Vs Test Accuracy")

listt= ["Train , Test"]

plt.bar( "Train", AccKNN_OnTrain_TFIDF, color ='maroon' ,width = 0.4)

plt.bar( "Test", AccKNN_OnTest_TFIDF, color ='b' , width = 0.4)

plt.show()


#---------------------------------------------------------Some Visualization-----------------------------------------------------

# TextDataPartitioning.Feature_Feature_Importance(np.array(X_trainBOW_PCA), np.array(y_train))
# TextDataPartitioning.Feature_Feature_Importance(np.array(X_trainTFIDF_PCA), np.array(y_train))
# TextDataPartitioning.Feature_Feature_Importance(np.array(X_trainNGRAM_PCA), np.array(y_train))

# -- Get Important Feature
TextDataPartitioning.plotImp(ClassifierXG_NGRAM , pd.DataFrame(X_trainNGRAM) , 20, (40, 20))

#Reduce Feature
ReducedFeature = pd.DataFrame(X_trainNGRAM).copy()

Important_Features =pd.DataFrame({'Value':ClassifierXG_NGRAM.feature_importances_,'Feature':ReducedFeature.columns}).sort_values(by="Value",  ascending=False)["Feature"][0:20]

#Drop Highst 20 important Features
ReducedFeature =  ReducedFeature.drop(Important_Features , axis = 1)

ReducedFeature_Test =  pd.DataFrame(X_testNGRAM).drop(Important_Features , axis = 1)


#--------------------------------------------------------- Train Champion Model on ReducedFeature -----------------------------------------------------

ClassifierKNN_NGRAM = TextDataPartitioning.KNN(ReducedFeature, y_train,5)

ReducedFeatureY_PredXG_NGRAM = TextDataPartitioning.Pred(ClassifierKNN_NGRAM,ReducedFeature_Test)



AccKNN_OnTrain_TFIDF = TextDataPartitioning.AccuracyTrain( ClassifierKNN_NGRAM , ReducedFeature, y_train)

AccKNN_OnTest_TFIDF  = TextDataPartitioning.AccuracyTest(y_test, ReducedFeatureY_PredXG_NGRAM)

#--------------------------------------------------------- PLot Train vs Test Accuracy on Reduced Feature Model -----------------------------------------------------


CF_MatrixKNN_TFIDF = TextDataPartitioning.ConfusionMatrix(y_test, ReducedFeatureY_PredXG_NGRAM)

TextDataPartitioning.PLOT_ConfusionMatrix(CF_MatrixKNN_TFIDF,'Champion Model')

sns.reset_defaults()
fig = plt.figure(figsize = (10, 5))

plt.title("champion Model Train Vs Test Accuracy")

listt= ["Train , Test"]

plt.bar( "Train", AccKNN_OnTrain_TFIDF, color ='maroon' ,width = 0.4)

plt.bar( "Test", AccKNN_OnTest_TFIDF, color ='b' , width = 0.4)

plt.show()

Reduced_KNN_Bais , Reduced_KNN_Vareis= TextDataPartitioning.BiasAndVariance(ReducedFeatureY_PredXG_NGRAM , y_test)


print("Drop Fist 40 important features from Data Set and TeTrain Our Model ")

print("-------------------")
print(f"Champion Model Train Accuracy : {AccKNN_OnTrain_TFIDF}")
print(f"Champion Model Test Accuracy : {AccKNN_OnTest_TFIDF}")

print(f"Champion Model Baise : {Reduced_KNN_Bais}")
print(f"Champion Model Varince : {Reduced_KNN_Vareis}")
print("-------------------")
print('\n\n')

