import os


class uitiles:
    DataSet_DF_columns=["filename","speaker","language","length","gender","accent","datasetname"]

    def Generate_File_Name(Language :str, Gender:str, I_Speaker:int, I_Clip:int,datasetname: str,Extention: str):
        return Language + "-" + Gender + "-" + str(I_Speaker) + "-" + str(I_Clip) + "-" + datasetname + "." + Extention

    def Save_To_CSV(PathToDF, df):
        if os.path.exists(PathToDF):
            df.to_csv(PathToDF, mode='a', header=False)
        else :
            df.to_csv(PathToDF)
