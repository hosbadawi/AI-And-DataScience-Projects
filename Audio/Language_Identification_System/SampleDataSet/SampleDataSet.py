import pandas as pd 
from os.path import exists


class SampleDataSet:
    
    '''
    To Take a sample from DataSet
    '''
    def _get_sample(self, df, n, random_State):
        sample_size =n 
        if sample_size > len(df):
            sample_size =len(df)
        return df.sample(sample_size,random_state=random_State, replace=False)


    def _get_sample_Above_Q(self, df, n, random_State, Q):
        sample_size =n 
        df = df[df["length"] >= df["length"].quantile(Q)]

        if sample_size > len(df):
            sample_size =len(df)
        return df.sample(sample_size,random_state=random_State, replace=False)

    def To_CSV(self, pathTo, df):
        df.to_csv(pathTo)

    def SampleDataBase(self, CSVPath, TotalNumber, MaxLength, MinLength=0, FileORTime="File", Random_State=42, ISMax=False):
        """Taking a sample from the dataset according to length 
            V 1.0.0
        Paramters:
                CSVPath str : Location to DataFrame

                TotalNumber int : the total number of required samples(Files,Minuts), this number must be divisible by 7 as each class will be the same size

                MaxLength double : Max length for each file in seconds

                MinLength double : Min length for each file in seconds

                FileORTime str :  Sample by file or time as TotalNumber will represent number of files or number of seconds
                                File for files OR seconds for time

                Random_State int : Random state from sampling

                ISMax bool : initial False it is select files with length between max and min

        Return:
            DataFrame With sampled files.
        
        """

        # Handeling the input Errors
        if TotalNumber % 7 !=0: raise Exception('TotalNumber Must be divisable by 7')
        if exists(CSVPath)==False : raise Exception("can't find the csv file in this path")

        #Each Class Count
        CountClass= TotalNumber//7
        
        #Read the DataFrame
        LanguageDf = pd.read_csv(CSVPath, low_memory=False) 
        LanguageDf.drop("Unnamed: 0", axis=1, inplace=True)

        if ISMax == True:
            LanguageDf=LanguageDf.loc[(LanguageDf["length"] >= MinLength) & (LanguageDf["length"] <= MaxLength )]

        #Shafful Dataset
        LanguageDf= LanguageDf.sample(frac=1,random_state=Random_State).reset_index(drop=True)

        #Check Length OR #Files
        if FileORTime == "File":
            LangFilesCount = LanguageDf.groupby("language")["language"].agg("count")
            OutOfRange = LangFilesCount[LangFilesCount < CountClass ]
            if len(OutOfRange!=0):
                    raise Exception(f"Can't Sample {CountClass} files from each class as the number is larger than existance files {OutOfRange}")
                
        elif FileORTime=="seconds":
            LangFilesCount = LanguageDf.groupby("language")["length"].agg("sum")
            OutOfRange = LangFilesCount[LangFilesCount < CountClass ]
            if len(OutOfRange!=0):
                    raise Exception(f"Can't Sample {CountClass} seconds from each class as the number is larger than existance files {OutOfRange}")


        #Final DataFrame
        Sampeld_DF =pd.DataFrame(columns= ["filename","speaker","language","length","gender","accent","datasetname"])


        #Sample based on number of files
        if FileORTime == "File":
            # loop over languages
            for language in LanguageDf["language"].unique():

                _CountClass=CountClass
                print(language)

                lang =LanguageDf.loc[LanguageDf["language"]==language]
                num_Speakers=len(lang["speaker"].unique())

                #Number of samples for every itteration
                n_files=5 
                while n_files * num_Speakers > _CountClass:
                    n_files=n_files-1

                if n_files <=0:
                      Sampeld_DF= pd.concat([Sampeld_DF,lang.sample(_CountClass, random_state=Random_State, replace=False)], ignore_index=True) 


                else:
                    
                    #itterate untill get number less than or equel 0 
                    while _CountClass - len(lang.groupby('speaker', group_keys=False).apply(lambda x: self._get_sample(x, n_files, Random_State))) >=0:
                        
                        #Take n Samples from each speaker
                        sample = lang.groupby('speaker', group_keys=False).apply(lambda x: self._get_sample(x, n_files, Random_State))

                        #Remove sampled rows from the DataFrame to not select it again
                        lang=lang[~lang.filename.isin(sample["filename"]) ]

                        #shafful dataset
                        lang= lang.sample(frac=1,random_state=Random_State).reset_index(drop=True)

                        #Append sampled dataframe to final dataframe                    
                        Sampeld_DF= pd.concat([Sampeld_DF,sample], ignore_index=True)

                        #reduce the number of remaining rows
                        _CountClass = _CountClass - len(sample)
                        
                        #Change the number of speakers as we droped sampled rows from initial dataset
                        num_Speakers=len(lang["speaker"].unique())

                        #calculate n_files again as 
                        while n_files * num_Speakers > _CountClass:
                            n_files=n_files-1

                        if n_files <=0:
                            Sampeld_DF= pd.concat([Sampeld_DF,lang.sample(_CountClass, random_state=Random_State, replace=False)], ignore_index=True)
                            break

                        if len(sample)==0:
                            break

            return Sampeld_DF


        #Sample based on sum of lengths
        elif FileORTime=="seconds":
            # loop over languages
            for language in LanguageDf["language"].unique():
                
                #Time For each class
                Time=CountClass

                print(language)

                lang =LanguageDf.loc[LanguageDf["language"]==language]

                num_Speakers=len(lang["speaker"].unique())

                #Number of samples for every itteration
                n_files=5 
                while n_files * num_Speakers * MaxLength > Time:
                    n_files=n_files-1
                            
                # if number of speakers is more than required sample rows
                if n_files <=0:
                    
                    while Time - Sampeld_DF.loc[Sampeld_DF["language"]==language,"length"].sum() > 0 :

                        # if len(lang[lang["length"]==max_length]) == 0:
                        #     max_length=lang["length"].max()

                        #Random sampleing from top 0.05 lengthes
                        sample  = lang[(lang["length"] >= lang["length"].quantile(0.95)) & (~lang["speaker"].isin(Sampeld_DF)) ].sample(1)

                        #Remove sampled rows from lang dataframe
                        lang=lang[~lang.filename.isin(sample["filename"]) ]
                        
                        #Shuffel dataset
                        lang= lang.sample(frac=1, random_state=42).reset_index(drop=True)

                        #Append to final dataframe
                        Sampeld_DF= pd.concat([Sampeld_DF,sample], ignore_index=True)

                            
                else:
                
                    while  Time -Sampeld_DF.loc[Sampeld_DF["language"]==language,"length"].sum()  > 0:
                        
                        #Take n Samples from top 0.02 lengths from each speaker
                        sample = lang.groupby('speaker', group_keys=False).apply(lambda x: self._get_sample_Above_Q(x,n_files,32,0.98))

                        #Remove sampled rows from dataframe
                        lang=lang[~lang.filename.isin(sample["filename"]) ]

                        #Shuffel dataFrame
                        lang= lang.sample(frac=1, random_state=32).reset_index(drop=True)

                        #append to final dataframe
                        Sampeld_DF= pd.concat([Sampeld_DF,sample], ignore_index=True)

                        #Update number of speakers as we removed sampled rows
                        num_Speakers=len(lang["speaker"].unique())

                        # reselected the best n_files based on new num_Speakers
                        while n_files * num_Speakers * MaxLength > Time - Sampeld_DF.loc[Sampeld_DF["language"]==language,"length"].sum() :
                            n_files=n_files-1

                        #if remains files number in lower than the #Speakers
                        if n_files <=0:
                            while Time - Sampeld_DF.loc[Sampeld_DF["language"]==language,"length"].sum() > 0 :

                                # if len(lang[lang["length"]==max_length]) == 0:
                                #     max_length=lang["length"].max()

                                sample  = lang[ lang["length"] >= lang["length"].quantile(0.98)].sample(1)

                                lang=lang[~lang.filename.isin(sample["filename"]) ]

                                lang= lang.sample(frac=1, random_state=32).reset_index(drop=True)

                                Sampeld_DF= pd.concat([Sampeld_DF,sample], ignore_index=True)    
                            break

            return Sampeld_DF      

# C = SampleDataSet().SampleDataBase("../../DataSets/FinalDataSet/FinalDataSet.csv", 100 *7, 8, 0, FileORTime="File") 
# SampleDataSet.To_CSV("Test100.csv", C)
#PASS