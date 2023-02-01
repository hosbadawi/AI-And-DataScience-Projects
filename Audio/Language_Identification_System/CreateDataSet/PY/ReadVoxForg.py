import tarfile
import pandas as pd 
import os
import shutil
import librosa
from IPython.display import clear_output
from uitiles import uitiles

class ReadVoxForg:
    """This Class aims to read tgz and unzip this files in folders within same language"""

    _Path ="../../DataSets/InititalVoxForge"

    def SetPath(self,Path):
        self._path=Path

    def ReadFileNames(self):
        """Read The text file contains files name and language
        
        Return: 
        DataFrame with FileNames and Languages
        """
        files = pd.read_csv(self._Path+"/voxforge_urls.txt",index_col=False, header= None ,  names=["FileNames"])
        filesNames_Lang = files["FileNames"].str.split('/', expand = True)[[2,7]]
        filesNames_Lang.columns=["Language","FileName"]
        filesNames_Lang=filesNames_Lang[filesNames_Lang["Language"]!='ru'] #As ru is not within the desired languages
        return filesNames_Lang

    def ReadFiles(self):   
        """Extact files in sperated folders based on the language""" 
        files = self.ReadFileNames() 
        Languages=  files["Language"].unique()
        ExtractedFiles = 0
        FailedFiles = 0
        for Language in Languages:
            for filename in files.loc[files["Language"]==Language,"FileName"]:
                try:
                    tar = tarfile.open(f'{self._Path}/{filename}', "r:*")
                    subdir_and_files = [
                            tarinfo for tarinfo in tar.getmembers()
                            if tarinfo.name.endswith(".wav")
                        ]
                    tar.extractall(members=subdir_and_files,path=f'../../DataSets/ExalVoxForge/{Language}')
                    tar.close()
                    ExtractedFiles=ExtractedFiles+1        
                except:
                    FailedFiles =FailedFiles+1
                    continue
        print(f"Extracted Files : {ExtractedFiles}")
        print(f"Field Files : {FailedFiles}")
    

    def MoveToOneFile(self,LangFilesName,Path,PathToMove,DataSetName,SaveToDF,PathToDF):
        """
        Move Specific langauges files in on folder

        Args:
            LangFilesName(list): list of folder names that contain the wav files

            Path(str):Path to languages files

            PathToMove(str):Path to Move

            DataSetName(str): Name of Dataset To create new name for each file
        """
        for  LangFile in LangFilesName:
            FilesNames =os.listdir(f"{Path}/{LangFile}")
            MovedFiles = []

            for I_Speaker, FilesName in enumerate(FilesNames) :
                
                print(f"File  : {I_Speaker} / {len(FilesNames)} - {LangFile}")
                splitedFilesName = FilesName.split('-')
                Speaker_Name=""
                if len(splitedFilesName) >=2 :
                   Speaker_Name= splitedFilesName[0] + "-" + splitedFilesName[1]
                else:
                    Speaker_Name= splitedFilesName[0] 

                lang = LangFile
                gender = "n"
                accent = ""
                for index, WavFile in enumerate (os.listdir(f"{Path}/{LangFile}/{FilesName}/wav")):
                    NewName= uitiles.Generate_File_Name(lang, gender, I_Speaker, index, DataSetName,"wav")  

                    try :
                        lenth =  librosa.get_duration(filename=f"{Path}/{LangFile}/{FilesName}/wav/{WavFile}")

                        shutil.copy2(f"{Path}/{LangFile}/{FilesName}/wav/{WavFile}", f'{PathToMove}/{NewName}')

                        MovedFiles.append([NewName,Speaker_Name,lang,lenth,gender,accent,DataSetName])
                    except BaseException as e:
                        print(f"Error While copy file {FilesName} - {WavFile} - ")
                        continue

                clear_output(wait=True)
        
            NewDF = pd.DataFrame(MovedFiles,columns=uitiles.DataSet_DF_columns)           
            if SaveToDF:
                uitiles.Save_To_CSV(PathToDF,NewDF)

            


