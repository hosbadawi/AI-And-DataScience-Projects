import nltk,re,pandas as pd,random,numpy as np
from nltk.corpus import stopwords
import datetime
from tqdm import tqdm
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error
from enchant.checker import SpellChecker
import os
import seaborn as sns

current_directory = os.getcwd()
print(current_directory) 
os.chdir("F:\hi")

def tweet_to_words(tweet):    
    # convert to lowercase
    text = tweet.lower()
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub("#bitcoin", 'bitcoin', text) # removes the '#' from bitcoin
    text = re.sub("#Bitcoin", 'Bitcoin', text) # removes the '#' from Bitcoin
    text = re.sub('#[A-Za-z0-9]+', '', text) # removes any string with a '#'
    text = re.sub('\\n', '', text) # removes the '\n' string
    text = re.sub('https:\/\/\S+', '', text) # removes any hyperlinks
    text = text.replace("#", "")
    text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', '', text, flags=re.MULTILINE)
    text = re.sub('@\\w+ *', '', text, flags=re.MULTILINE)
    # tokenize
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    words = [PorterStemmer().stem(w) for w in words]
    # return list
    return words

def unlist(list):
    words=''
    for item in list:
        words+=item+' '
    return words

def compute_vader_scores(df, label):
    sid = SentimentIntensityAnalyzer()
    df["vader_neg"] = df[label].apply(lambda x: sid.polarity_scores(unlist(x))["neg"])
    df["vader_neu"] = df[label].apply(lambda x: sid.polarity_scores(unlist(x))["neu"])
    df["vader_pos"] = df[label].apply(lambda x: sid.polarity_scores(unlist(x))["pos"])
    df["vader_comp"] = df[label].apply(lambda x: sid.polarity_scores(unlist(x))["compound"])
    df['cleantext2'] = df[label].apply(lambda x: unlist(x))
    return df


# Read data and format Date column and sort date
Fulldata = pd.read_csv('tweets.csv',sep=';', usecols=['timestamp','text'])
Fulldata.columns = ['Dates', 'Tweets']
Fulldata['Dates'] = pd.to_datetime(Fulldata['Dates'],format="%Y-%m-%d").dt.date
Fulldata = Fulldata.sort_values(by=['Dates'], ascending=True)
Fulldata = Fulldata[Fulldata["Dates"] >= datetime.datetime.strptime('2014-01-01', "%Y-%m-%d").date()]

# get number of tweets for each unique date
Num_of_tweets = Fulldata['Dates'].value_counts()
Num_of_tweets = pd.DataFrame(Num_of_tweets)
Num_of_tweets.reset_index(inplace=True)
Num_of_tweets = Num_of_tweets.rename(columns = {'index':'Dates'})
Num_of_tweets.columns = ['Dates', 'NumOfTweets']
Num_of_tweets = Num_of_tweets.sort_values(by=['Dates'], ascending=True)
Num_of_tweets.reset_index(inplace=True, drop=True)

# get the first 150 tweets for each unique dates
Fulldata_Copy = Fulldata.copy()
Fulldata_Copy.columns = ['Dates', 'Tweets']
Fulldata_Copy = Fulldata_Copy.sort_values(by=['Dates'], ascending=True)
Fulldata_Copy.reset_index(inplace=True, drop=True)

print(Num_of_tweets['NumOfTweets'].min()) # print the minimum number of tweets
Dataset = pd.DataFrame() # empty dataframe to append the values of 150 tweets for each unique date

UniqueDates = list(Num_of_tweets['Dates'])

# get the index of unique dates
index_of_unique_dates = []
for i in range(len(UniqueDates)):
    temp = Fulldata_Copy.index[Fulldata_Copy['Dates'] == UniqueDates[i]].tolist()
    index_of_unique_dates.append(temp[0])
del temp

# take the first 150 tweet for each unique Date
Fulldata_Copy['Tweets'] = Fulldata_Copy['Tweets'].astype(str)
for i in range(len(UniqueDates)):
    temp = pd.DataFrame()
    temp = Fulldata_Copy.iloc[index_of_unique_dates[i]:index_of_unique_dates[i]+150,:]
    Dataset = Dataset.append(temp)
        
Dataset.reset_index(inplace=True, drop=True)
Dataset['Tweets'] = Dataset['Tweets'].astype(str)
Dataset = Dataset[Dataset["Dates"] >= datetime.datetime.strptime('2014-09-16', "%Y-%m-%d").date()]


# Detect the language of the tweets
languageList = []
max_error_count = 5
min_text_length = 3
for i in tqdm(range(len(Dataset['Tweets']))):
    d = SpellChecker("en_US")
    d.set_text(Dataset['Tweets'][i])
    errors = [err.word for err in d]
    if ((len(errors) > max_error_count) or len(Dataset['Tweets'][i].split()) < min_text_length):
        languageList.append(False)
    else:
        languageList.append(True)

indexOfNotEng = []
indexOfNotEng = [i for i, x in enumerate(languageList) if x == False]
a = Dataset.copy()
a.drop(indexOfNotEng, axis=0, inplace=True)
a.reset_index(inplace=True, drop=True)

# Clean the tweets and vectorize it
cleantext=[]
for item in tqdm(a['Tweets']):
    words = tweet_to_words(item)
    cleantext+=[words]
a['cleantext'] = cleantext

# compute vader scores
cpyDataset = pd.DataFrame()
cpyDataset = compute_vader_scores(a,'cleantext')

# Prepare the final DataSet
FinalData = cpyDataset.copy()
FinalData = FinalData.drop(['Tweets', 'cleantext', 'cleantext2'], axis=1)
FinalData.reset_index(inplace=True, drop=True)
FinalData = FinalData.groupby(FinalData['Dates'])['vader_neg','vader_neu','vader_pos','vader_comp'].agg(lambda x: x.unique().mean())
FinalData.reset_index(inplace=True)

BitcoinPrices = pd.read_csv('BTC-USD.csv')
BitcoinPrices = BitcoinPrices.drop(['Open', 'High', 'Low','Close'], axis=1)
BitcoinPrices['Date'] = BitcoinPrices['Date'].astype(str)

for i in tqdm(range(len(BitcoinPrices))):
    BitcoinPrices['Date'][i] = datetime.datetime.strptime(BitcoinPrices['Date'][i], "%Y-%m-%d").date()

BitcoinPrices.drop([529 , 1416], axis=0, inplace=True)
BitcoinPrices.reset_index(inplace=True, drop=True)


df = pd.concat([pd.DataFrame(FinalData['Dates']), pd.DataFrame(BitcoinPrices['Date'])],axis=1 , ignore_index = True)
df.columns = ['final','bit']
df['final'] = df['final'] + pd.to_timedelta(1,unit='D')
ls = df['final']==df['bit']
del df,ls


# df = df['bit'].sort_values(na_position='last')
# df.reset_index(inplace=True, drop=True)
# df = pd.concat([pd.DataFrame(FinalData['Dates']), pd.DataFrame(df)],axis=1 , ignore_index = True)
# df.columns = ['final','bit']
# df['final'] = df['final'] + pd.to_timedelta(1,unit='D')


FinalData = pd.concat([pd.DataFrame(FinalData), pd.DataFrame(BitcoinPrices)],axis=1 , ignore_index = True)
FinalData = FinalData.drop([0], axis=1)
FinalData.columns = ['Neg', 'Neu','Pos','Comp','Date','Bitcoin_Price','Volume']

X = FinalData.iloc[:,[0,1,2,3,6]]
Y = FinalData.iloc[:,[4,5]]

X.to_csv(r'F:\\hi\\X.csv', index = False, header = True)
Y.to_csv(r'F:\\hi\\Y.csv', index = False, header = True)

X = pd.read_csv('X.csv')
Y = pd.read_csv('Y.csv')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

X_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
Y_train.reset_index(inplace=True, drop=True)
Y_test.reset_index(inplace=True, drop=True)

dateTest = Y_test.iloc[:,0]

# dateTrain = Y_train.iloc[:,0]

Y_train = Y_train.drop('Date', axis=1)
Y_test = Y_test.drop('Date', axis=1)

bitcoinTest = pd.DataFrame()
bitcoinTest = pd.concat([pd.DataFrame(dateTest), pd.DataFrame(Y_test)],axis=1 , ignore_index = True)
bitcoinTest = bitcoinTest.sort_values(by=[0], ascending=True)
bitcoinTest.reset_index(inplace=True, drop=True)

# Linear 
reg = LinearRegression().fit(X_train, Y_train)
Y_pred = reg.predict(X_test)
reg.score(X_test,Y_test)

# Plot Linear Regression
linearTest = pd.concat([pd.DataFrame(dateTest), pd.DataFrame(Y_pred)],axis=1 , ignore_index = True)
linearTest = linearTest.sort_values(by=[0], ascending=True)
linearTest.reset_index(inplace=True, drop=True)
linearTest.columns = ['Date', 'Predicted']

mean_squared_error(Y_test,Y_pred)

plt.plot(bitcoinTest.iloc[:,0], bitcoinTest.iloc[:,1], color='#00A093', label = 'Bitcoin Prices')
plt.plot(linearTest.iloc[:,0], linearTest.iloc[:,1], color='#FF0000', label = 'Linear Regression')
plt.legend(fontsize=7)
plt.xlabel('Time', fontsize = 15)
plt.ylabel('Prices', fontsize = 15)
positions = (0,28,148,276,391, 518,640)
labels = ("2014",'2015','2016','2017',"2018", "2019","2020")
plt.xticks(positions, labels)
plt.show()

# # create a regressor object
# ll= []
# for i in tqdm(range(10000)):
#     regressortree = DecisionTreeRegressor(random_state = 6572) 
#     # fit the regressor with X and Y data
#     regressortree = regressortree.fit(X_train, Y_train)
#     # ypredTree = regressortree.predict(X_test)
#     ll.append(regressortree.score(X_test,Y_test))

#create a regressor object
regressortree = DecisionTreeRegressor(random_state = 6572) 
# fit the regressor with X and Y data
regressortree = regressortree.fit(X_train, Y_train)
ypredTree = regressortree.predict(X_test)
regressortree.score(X_test,Y_test)

TreeTest = pd.concat([pd.DataFrame(dateTest), pd.DataFrame(ypredTree)],axis=1 , ignore_index = True)
TreeTest = TreeTest.sort_values(by=[0], ascending=True)
mean_squared_error(Y_test,ypredTree)

plt.plot(bitcoinTest.iloc[:,0], bitcoinTest.iloc[:,1], color='#00A093', label = 'Bitcoin Prices')
plt.plot(TreeTest.iloc[:,0], TreeTest.iloc[:,1], color='#FFA500', label = 'Decision Tree Regression')
plt.legend(fontsize=7)
plt.xlabel('Time', fontsize = 15)
plt.ylabel('Prices', fontsize = 15)
positions = (0,28,148,276,391, 518,640)
labels = ("2014",'2015','2016','2017',"2018", "2019","2020")
plt.xticks(positions, labels)
plt.show()

# SVR
from sklearn.svm import SVR
regressorSVR = SVR(C=800.0, epsilon=0.5)
regressorSVR = regressorSVR.fit(X_train, Y_train.squeeze())
ypredSVR = regressorSVR.predict(X_test)
regressorSVR.score(X_test,Y_test)

SVRTest = pd.concat([pd.DataFrame(dateTest), pd.DataFrame(ypredSVR)],axis=1 , ignore_index = True)
SVRTest = SVRTest.sort_values(by=[0], ascending=True)
mean_squared_error(Y_test,ypredSVR)

plt.plot(bitcoinTest.iloc[:,0], bitcoinTest.iloc[:,1], color='#00A093', label = 'Bitcoin Prices')
plt.plot(SVRTest.iloc[:,0], SVRTest.iloc[:,1], color='#ff76d8', label = 'SVR')
plt.legend(fontsize=7)
plt.xlabel('Time', fontsize = 15)
plt.ylabel('Prices', fontsize = 15)
positions = (0,28,148,276,391, 518,640)
labels = ("2014",'2015','2016','2017',"2018", "2019","2020")
plt.xticks(positions, labels)
plt.show()


# random cutforest

# # Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
# ll = []
# for i in tqdm(range(5,100)):
#     regressorRF = RandomForestRegressor(n_estimators = i, random_state = 613)
#     regressorRF.fit(X_train, Y_train)
#     ll.append(regressorRF.score(X_test,Y_test))

# Fitting Random Forest Regression to the dataset
regressorRF = RandomForestRegressor(n_estimators = 9, random_state = 613)
regressorRF.fit(X_train, Y_train)
y_predRF = regressorRF.predict(X_test)
regressorRF.score(X_test,Y_test)

RFTest = pd.concat([pd.DataFrame(dateTest), pd.DataFrame(y_predRF)],axis=1 , ignore_index = True)
RFTest = RFTest.sort_values(by=[0], ascending=True)
mean_squared_error(Y_test,y_predRF)


plt.plot(bitcoinTest.iloc[:,0], bitcoinTest.iloc[:,1], color='#00A093', label = 'Bitcoin Prices')
plt.plot(RFTest.iloc[:,0], RFTest.iloc[:,1], color='#0300A7', label = 'RandomForest')
plt.legend(fontsize=7)
plt.xlabel('Time', fontsize = 15)
plt.ylabel('Prices', fontsize = 15)
positions = (0,28,148,276,391, 518,640)
labels = ("2014",'2015','2016','2017',"2018", "2019","2020")
plt.xticks(positions, labels)
plt.show()