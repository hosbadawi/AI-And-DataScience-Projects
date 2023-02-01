import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
#######################################################################################
# data loading and splitting
df = pd.read_csv('MCSDatasetNEXTCONLab.csv')
x = df.iloc[:, 0:12]
y = df.iloc[:, 12]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = x_train.reset_index(drop=True), x_test.reset_index(drop=True),y_train.reset_index(drop=True), y_test.reset_index(drop=True)

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
    
# models fitting and prediction
def models(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = x_train.reset_index(drop=True), x_test.reset_index(drop=True),y_train.reset_index(drop=True), y_test.reset_index(drop=True)
    model = model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    report_test = classification_report(y_test, y_test_pred)
    return (model, y_train_pred, y_test_pred, accuracy_train, accuracy_test, report_test)

#---------------------------------------------MAIN---------------------------------

RF, AB, NB = RandomForestClassifier(), AdaBoostClassifier(), GaussianNB()

#first model (Random Forest)
model_RF, y_train_pred_RF, y_test_pred_RF, accuracy_train_RF, accuracy_test_RF, report_RF = models(RF, x, y)
cf_RF = ConfusionMatrix(y_test, y_test_pred_RF)
PLOT_ConfusionMatrix(cf_RF,'RF Confusion Matrix')

#second model (Adaboost)
model_AB, y_train_pred_AB, y_test_pred_AB, accuracy_train_AB, accuracy_test_AB, report_AB = models(AB, x, y)
cf_AB = ConfusionMatrix(y_test, y_test_pred_AB)
PLOT_ConfusionMatrix(cf_AB,'AB Confusion Matrix')

#third model (Naive Bayes)
model_NB, y_train_pred_NB, y_test_pred_NB, accuracy_train_NB, accuracy_test_NB, report_NB = models(NB, x, y)
cf_NB = ConfusionMatrix(y_test, y_test_pred_NB)
PLOT_ConfusionMatrix(cf_NB,'NB Confusion Matrix')

#frist ensemble framework : majority voting-based aggregator
y_pred_Voting = pd.concat([pd.DataFrame(y_test_pred_RF), pd.DataFrame(y_test_pred_AB), pd.DataFrame(y_test_pred_NB)],axis=1 , ignore_index = True)
y_pred_Voting = y_pred_Voting.sum(axis = 1)
y_pred_Voting = y_pred_Voting.replace(1, 0)
y_pred_Voting = y_pred_Voting.replace(3, 1)
y_pred_Voting = y_pred_Voting.replace(2, 1)
cf_Voting = ConfusionMatrix(y_test, y_pred_Voting)
PLOT_ConfusionMatrix(cf_Voting,'Voting Confusion Matrix')
report_Voting = classification_report(y_test, y_pred_Voting)
accuracy_Voting = accuracy_score(y_test, y_pred_Voting)

#second ensemble framework : weighted sum aggregation
total = accuracy_train_RF + accuracy_train_AB + accuracy_train_NB
w_RF, w_AB, w_NB = accuracy_train_RF/total, accuracy_train_AB/total, accuracy_train_NB/total
aggregated_output = (w_RF * y_test_pred_RF) + (w_AB * y_test_pred_AB) + (w_NB * y_test_pred_NB)

y_pred_weighted_sum  = []
for i in aggregated_output: 
    if i > 0.5: y_pred_weighted_sum.append(1)
    else: y_pred_weighted_sum.append(0)
    
y_pred_weighted_sum = pd.DataFrame(y_pred_weighted_sum).squeeze()
accuracy_weighted_sum = accuracy_score(y_test, y_pred_weighted_sum)
report_weighted_sum = classification_report(y_test, y_pred_weighted_sum)
cf_weighted_sum = ConfusionMatrix(y_test, y_pred_weighted_sum)
PLOT_ConfusionMatrix(cf_weighted_sum,'weighted_sum Confusion Matrix')

#Bar Plots
Bars = ['RF','AB','NB','Voting', 'weighted_sum']
colors = ['#EFC7C2' , '#FFE5D4','#8D86C9','#68A691','#694F5D']
accuracies = [accuracy_test_RF,accuracy_test_AB,accuracy_test_NB,accuracy_Voting, accuracy_weighted_sum]
plt.bar(Bars, accuracies, 0.4, color = colors )
plt.xlabel("5 Models")
plt.ylabel('Accuracies on Test')
plt.ylim(0.7,1)
plt.title("Accuracies comparison")