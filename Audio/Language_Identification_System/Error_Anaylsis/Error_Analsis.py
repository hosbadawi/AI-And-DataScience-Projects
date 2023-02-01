from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PLot_CR import PLot_CR

class Error_anaylsis():
    def __init__(self, y_true, y_pred, classes, PathToSavePLots = "", Df_Path="", file_names= None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes
        self.save_path = PathToSavePLots
        self.Df_Path=Df_Path
        self.file_names=file_names
        self.caculted_TF_DF=False
        self.TF_DF= None
    
    def CM(self, plotTitle, Saving=True):
        """Plot the confusion Matrix
            Paramters :

            plotTitle(str): The plot title the diffule "CR"

            Saving(Bool): save the plot in the given path the diffult False
        """
        result = confusion_matrix(self.y_true,  self.y_pred)
        df_cm = pd.DataFrame(result, index = [i for i in self.classes],
                      columns = [i for i in self.classes])
        plt.figure(figsize = (10,7))
        plt.title(plotTitle)
        sns.heatmap(df_cm, annot=True, fmt='g')
        
        if Saving ==True:
            plt.savefig(f"{self.save_path}/{plotTitle}.jpg")
        plt.show()

        
    def CR(self, printing=True, ploting=False, plotTitle="CR", Saving=False ):
        """Calculating the Classification Report and plot it  
        
            Paramters :
            
            printing(Bool): Print the CR the Deffult True
            
            ploting(Bool): Ploting the the CR diffulte False
            
            plotTitle(str): The plot title the diffulte "CR"
            
            Saving(Bool): save the plot in the given path the diffult False
        """
        cr = classification_report(self.y_true,  self.y_pred, target_names=self.classes)
        if printing == True:
            print(cr)
            
        if ploting == True:
            PLot_CR().plot_classification_report(cr, self.classes, 7, title=plotTitle, save = True, save_path=self.save_path )
   
    
    def Create_TF_DF(self):
        if(self.file_names)==None:
            raise Exception("you shouid enter files names in the order of the predictions")
        
        True_OF_False = (self.y_pred == self.y_true).astype(int).tolist()
        self.TF_DF = pd.DataFrame({"filename" : self.file_names, "T_F" : True_OF_False})
        
        Originaldf = pd.read_csv(self.Df_Path)
        Originaldf =  Originaldf[Originaldf["filename"].isin(self.TF_DF["filename"].values)]
        
        self.TF_DF = pd.merge( Originaldf, self.TF_DF, on=["filename"])
        self.caculted_TF_DF =True
        
    
    def All_Length_disPlot(self, PLotTorF=1, plotTitle="Length_disPlot", saveing=False, plotType="kde"):
        """Plot distripution plot for the audios duiorations all for all languages in one figure
        
            PLotTorF(int): get the most speakers the model get them true if it is ==1 or the most flase ==0 diffult ==1
            
            plotTitle(str): The plot title the diffulte "Speakers_disPlot"

            saveing(Bool): save the plot in the given path the diffulte False

            plotType(str): the ploting type for displot {“hist”, “kde”, “ecdf”} the diffulte kde
        """
        
        if self.caculted_TF_DF==False:
            self.Create_TF_DF()
            
        sns.displot(data = self.TF_DF[self.TF_DF.T_F==PLotTorF], x="length", kind=plotType, hue="language" ).set(title=plotTitle)
        if saveing ==True:
            plt.savefig(f"{self.save_path}/{plotTitle}.jpg")
            plt.show()  
            
            
            
            
    def Individually_Length_disPlot(self, PLotTorF=1, plotTitle="Length_disPlot", saveing=False, plotType="kde", langs="All", Train=False):
        """Plot distripution plot for the audios duiorations all for each language in langs in one figure for each language 
        
            PLotTorF(int): get the most speakers the model get them true if it is ==1 or the most flase ==0 diffult ==1
            
            plotTitle(str): The plot title the diffulte "Speakers_disPlot"

            saveing(Bool): save the plot in the given path the diffulte False

            plotType(str): the ploting type for displot {“hist”, “kde”, “ecdf”} the diffulte kde
            
            langs(list): the languages for ploting the diffulte all which wil plot all languages
            
            Train(Bool): Include the train data into the graph or not diffult False
        """
        
        
        if self.caculted_TF_DF==False:
            self.Create_TF_DF()
            
        if langs =="All":
            langs= [lang for lang in self.classes]
            
        if PLotTorF==1:
            Legend_Name= "True Predicted"
        else:
            Legend_Name= "False Predicted"
            
        if Train==True:
            Traindf = pd.read_csv(self.Df_Path)
            
            
        for lang in langs:
            dataPredicted = self.TF_DF[(self.TF_DF.T_F==PLotTorF) & (self.TF_DF.language==lang)]
            dataTest = self.TF_DF[self.TF_DF.language==lang]
            dataTrain = Traindf[Traindf.language==lang]

            if Train==True:
                ax = dataTrain.length.plot(kind = 'density',  alpha = 0.9, color="b", label = "Train", linewidth=2)
                
            ax = dataTest.length.plot(kind = 'density',  alpha = 0.4,color="g", label = 'Test', linewidth=4)
            ax = dataPredicted.length.plot(kind = 'density',  alpha = 0.7, color="r", label = Legend_Name, linewidth=2)

            
            ax.set(xlabel="Length", ylabel="density")
            ax.set(title=f"{plotTitle}_{lang}")
            ax.legend(fontsize=8)
            if saveing ==True:
                plt.savefig(f"{self.save_path}/{plotTitle}_{lang}.jpg")
            plt.show()
            
            
    def All_Speakers_disPlot(self, PLotTorF=1, plotTitle="Speakers_disPlot", saveing=False, Train=False, TopK=5):
        """Plot barPLot figure shows the most Speakers the model get them true of false based on the passed paramters
        
            PLotTorF(int): get the most Speakers the model get them true if it is ==1 or the most flase ==0 diffult ==1
            
            plotTitle(str): The plot title the diffulte "Speakers_disPlot"

            saveing(Bool): save the plot in the given path the diffult False

            Train(Bool): Include the train data into the graph or not diffult False
            
            TopK(int): get the top k speakers diffult 5
            
        """
        
        
        if self.caculted_TF_DF==False:
            self.Create_TF_DF()
            
        if PLotTorF==1:
            Legend_Name= "True Predicted"
            color = "forestgreen"
        else:
            Legend_Name= "False Predicted"
            color = "darkred"
   
        All_Test = self.TF_DF.groupby("speaker")["T_F"].agg("count")
        Predicted = self.TF_DF[self.TF_DF["T_F"]==PLotTorF].groupby("speaker")["T_F"].agg("count").sort_values(ascending=False)[0:TopK]   
        Renamed_Speakrs = [i+1 for i in range(len(Predicted))]
        
        if Train==False:
             # Test
            DF_ = pd.DataFrame(
                     data = { "Renamed_Speakrs": Renamed_Speakrs ,
                         "values" :  All_Test[Predicted.index].values,
                         "type": ["Test"] *len(Predicted[Predicted.index].values) })
            

            # Predicted
            DF_ = pd.concat([DF_ ,
                        pd.DataFrame(data = { "Renamed_Speakrs": Renamed_Speakrs ,
                         "values" : Predicted.values,
                         "type": [Legend_Name] *len(Predicted.values) })])
            
            ax = sns.barplot(data=DF_, x="Renamed_Speakrs", y="values", hue="type",  ci=None, errwidth=0, palette=["gray", color])
            
            ax.bar_label(ax.containers[0])
            ax.bar_label(ax.containers[1])
            ax.set(xlabel="Speaker", ylabel="Count")
            ax.set(title=plotTitle)
            
            
        else:
            Traindf = pd.read_csv(self.Df_Path)
            TrainSPCount = Traindf.groupby("speaker")["speaker"].agg("count")
            
            # Train
            DF_ = pd.DataFrame(
                        data = { "Renamed_Speakrs": Renamed_Speakrs ,
                         "values" : TrainSPCount[Predicted.index].values,
                         "type": ["Train"] *len(TrainSPCount[Predicted.index].values) })
            
            
            # Test
            DF_ = pd.concat([DF_ ,pd.DataFrame(
                        data = { "Renamed_Speakrs": Renamed_Speakrs ,
                         "values" : All_Test[Predicted.index].values,
                         "type": ["Test"] *len(Predicted[Predicted.index].values) })])
            
            
            # Predicted
            DF_ = pd.concat([DF_ ,pd.DataFrame(
                        data = { "Renamed_Speakrs": Renamed_Speakrs ,
                         "values" : Predicted.values,
                         "type": [Legend_Name] *len(Predicted.values) })])
            
            
            ax = sns.barplot(data=DF_, x="Renamed_Speakrs", y="values", hue="type",  ci=None, errwidth=0, palette=["C0", "gray", color])
            ax.bar_label(ax.containers[0])
            ax.bar_label(ax.containers[1])
            ax.bar_label(ax.containers[2])


            ax.set(xlabel="Speaker", ylabel="Count")
            ax.set(title=plotTitle)
        if saveing ==True:
            plt.savefig(f"{self.save_path}/{plotTitle}.jpg")
        plt.show()
        
        
        

    def All_Gender_disPlot(self, PLotTorF=1, plotTitle="Speakers_disPlot", saveing=False, Train=False, Gander="All"):
        """Plot barPLot figure shows the most speakers the model get them true of false based on the passed paramters

            PLotTorF(int): get the most speakers the model get them true if it is ==1 or the most flase ==0 diffult ==1

            plotTitle(str): The plot title the diffulte "Speakers_disPlot"

            saveing(Bool): save the plot in the given path the diffult False

            Train(Bool): Include the train data into the graph or not diffult False
            
            Gander(list): determine the ploted gender diffult value is all == ['n', 'm', 'f']
        """


        if self.caculted_TF_DF==False:
            self.Create_TF_DF()
        if PLotTorF==1:
            Legend_Name= "True Predicted"
            color = "forestgreen"
        else:
            Legend_Name= "False Predicted"
            color = "darkred"
         
        if  Gander=="All":
            Gander =['n', 'm', 'f']
            
            

        All_Test = self.TF_DF.groupby("gender")["T_F"].agg("count")
        Predicted = self.TF_DF[self.TF_DF["T_F"]==PLotTorF].groupby("gender")["T_F"].agg("count").sort_values(ascending=False) 
        Gender_Name = [i for i in Predicted.index]
        
        # PLot Test and Predicted
        if Train==False:
             # Test
            DF_ = pd.DataFrame(
                     data = { "Gender": Gender_Name ,
                         "values" :  All_Test[Predicted.index].values,
                         "type": ["Test"] *len(Predicted[Predicted.index].values) })


            # Predicted
            DF_ = pd.concat([DF_ ,
                        pd.DataFrame(data = { "Gender": Gender_Name ,
                         "values" : Predicted.values,
                         "type": [Legend_Name] *len(Predicted.values) })])

            DF_ = DF_[DF_.Gender.isin(Gander)] 
            print(DF_)
            ax = sns.barplot(data=DF_, x="Gender", y="values", hue="type",  ci=None, errwidth=0, palette=["gray", color])

            ax.bar_label(ax.containers[0])
            ax.bar_label(ax.containers[1])
            ax.set(xlabel="Gender", ylabel="Count")
            ax.set(title=plotTitle)

        # PLot Train, Test and Predicted
        else:
            Traindf = pd.read_csv(self.Df_Path)
            TrainSPCount = Traindf.groupby("gender")["gender"].agg("count")

            # Train
            DF_ = pd.DataFrame(
                        data = { "Gender": Gender_Name ,
                         "values" : TrainSPCount[Predicted.index].values,
                         "type": ["Train"] *len(TrainSPCount[Predicted.index].values) })


            # Test
            DF_ = pd.concat([DF_ ,pd.DataFrame(
                        data = { "Gender": Gender_Name ,
                         "values" : All_Test[Predicted.index].values,
                         "type": ["Test"] *len(Predicted[Predicted.index].values) })])


            # Predicted
            DF_ = pd.concat([DF_ ,pd.DataFrame(
                        data = { "Gender": Gender_Name ,
                         "values" : Predicted.values,
                         "type": [Legend_Name] *len(Predicted.values) })])

            DF_ = DF_[DF_.Gender.isin(Gander)] 
            ax = sns.barplot(data=DF_, x="Gender", y="values", hue="type",  ci=None, errwidth=0, palette=["C0", "gray", color])
            ax.bar_label(ax.containers[0])
            ax.bar_label(ax.containers[1])
            ax.bar_label(ax.containers[2])


            ax.set(xlabel="Gender", ylabel="Count")
            ax.set(title=plotTitle)
        if saveing ==True:
            plt.savefig(f"{self.save_path}/{plotTitle}.jpg")
        plt.show()
