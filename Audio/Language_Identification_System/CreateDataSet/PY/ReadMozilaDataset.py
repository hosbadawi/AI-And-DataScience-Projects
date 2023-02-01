import pandas as pd
import pydub
import os
from IPython.display import clear_output
from uitiles import uitiles

class ReadMozilaDataset:
    def __init__(self,ReadTSVPath,ReadClipsPath,SavePath,DataSetName):
        """
        Read mb3 files from Mozila common voice and convert them to WAV

        args :
            ReadTSVPath(str) : path to Folder that contains Train.tsv, Test.tsv, and dev.tsv files
            ReadClipsPath(str) : path to Folder that contains.mb3 clips
            SavePath(str) : path to Folder to save converted files in it
            DataSetName(str) : Will used to build new name for converted file
        """
        self.ReadTSVPath=ReadTSVPath
        self.ReadClipsPath=ReadClipsPath
        self.SavePath=SavePath
        self.DataSetName=DataSetName
        self.DataInitialDF= self.ReadTSVFiles()

    def ReadTSVFiles(self):
        """Read Train, Test, and dev files and Concat them in one dataFrame
        
        args :
            FolderPath(str) : path to Folder that contains Train.tsv, Test.tsv, and dev.tsv files
        
        return :
             concated DataFrame with data from Train, Test, and dev files

        """
        train = pd.read_csv(self.ReadTSVPath+'/train.tsv', sep='\t')
        test = pd.read_csv(self.ReadTSVPath+'/test.tsv', sep='\t')
        dev = pd.read_csv(self.ReadTSVPath+'/dev.tsv', sep='\t')
        
        return pd.concat([train, test ,dev], ignore_index=True)
         

    def Convert(self, SaveToDF=False,PathToSave=""):
        """ 
        Using Concated DataFrame to Read mb3 files and convert it and return new DF with converted files

        args:
            SaveToDF : To save to converted files metadate to .csv file or add to exist .csv file, defulat=False 

            PathToSave : Path to save csv file or to add to exist csv fill NOTE:You must enter path with filename.csv example ../test.csv
        """

        Speakers =self.DataInitialDF["client_id"].unique()
        ConvertedFiles = []
        for I_Speaker, Speaker in enumerate(Speakers):
            SpeakerClips = self.DataInitialDF[self.DataInitialDF["client_id"]==Speaker]
            print(f"I_speaker  : {I_Speaker} / {len(Speakers)}")

            for index, clip in SpeakerClips.iterrows():
                lang = clip["locale"]   
                gender =  str(clip["gender"])[0]  if str(clip["gender"])[0] !='o' else 'n'
                accent =  str(clip["accents"])
                NewName= uitiles.Generate_File_Name(lang, gender, I_Speaker, index, self.DataSetName,"wav")  
                try:
                    sound = pydub.AudioSegment.from_mp3(f"{self.ReadClipsPath}/{clip['path']}")
                    sound.export(f"{self.SavePath}/{NewName}", format="wav") 
                    ConvertedFiles.append([NewName,Speaker,lang,sound.duration_seconds,gender,accent,self.DataSetName])
                except:
                    print(f"Error while converting fille {clip['path']}")
                    continue
            clear_output(wait=True)
            
        NewDF = pd.DataFrame(ConvertedFiles,columns=uitiles.DataSet_DF_columns)           
        if SaveToDF:
            uitiles.Save_To_CSV(PathToSave,NewDF)


        return NewDF

# ar = ReadMozilaDataset("../../DataSets/ExalVoxForge/mozila-ar","../../DataSets/ExalVoxForge/mozila-ar/clips","../../DataSets/FinalDataSet/clips","CV")
# ar.Convert(True,"../../DataSets/FinalDataSet/FinalDataSet.csv")
