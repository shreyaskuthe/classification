import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
#%%
adult_df=pd.read_csv(r'E:\DATA SCIENCE\imarticus\python\datasets\adult_data.csv',header=None,delimiter=' *, *')
print(adult_df.head()) ##delimiter is used to rmove spaces if there are any space present in data set
#%% to display all headers in spydeer
pd.set_option('display.max_columns',None)
adult_df.head()
#%%
adult_df.shape
#%% to give column names
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

adult_df.head()
#%% pre processing of data 1)feature selection 2)handeling missing values 
##creat the copy of dataframe
adult_df_rev=pd.DataFrame.copy(adult_df)
# axis =1 to drop column axis =0 drop row
adult_df_rev=adult_df_rev.drop(["education","fnlwgt"],axis=1)
adult_df_rev.isnull().sum()
#%%[] used to pass multiple characters to replace values
adult_df_rev.dtypes
adult_df_rev=adult_df_rev.replace(['?'],np.nan)
adult_df_rev.isnull().sum()
#%% replcing missing values with mode values
# inplace = true used to avoid assignment afterwords
for value in ['workclass','occupation','native_country']:
    adult_df_rev[value].fillna(adult_df_rev[value].mode()[0],inplace=True)
#%%
adult_df_rev.workclass.mode()[0]
#%%
adult_df_rev.isnull().sum()
#%% generic code to replace missing value without entering column name manually 
"""
for x in adult_df_rev.columns[:]:
        if adult_df_rev[x].dtype=='object':
        adult_df_rev[x].fillna(adult_df_rev[x].mode()[0],inplace=True)
        elif adult_df_rev[x].dtype=='int64' or adult_df_rev[x].dtype=='float64':
        adult_df_rev[x].fillna(adult_df_rev[x].mean(),inplace=True)
"""
#%% to get all the charatere data
colname=[]
for x in adult_df_rev.columns:
    if adult_df_rev[x].dtype=='object':
        colname.append(x)
colname
#%% label encoding for pre processing of data to convert char\obj to numeric
from  sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname:
        adult_df_rev[x]=le.fit_transform(adult_df_rev[x])
adult_df_rev.head()
#%%
#0--> <=50k
#1--> >50k
#%%
X=adult_df_rev.values[:,:-1]
Y=adult_df_rev.values[:,-1]
#%% scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
print(X)
#%%
Y=Y.astype(int)
#%% 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
#%%
from sklearn.linear_model  import LogisticRegression 
#creat model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))
print(classifier.coef_)
print(classifier.intercept_)
#%%  model evaluation
#recall -accuracy score for individual classes (choose recall bet recall and precision )
#specificity ( true negative rate\recall for class 0)
#how many of the negative cases didn we catch?(formula=TN\TN+FP)
#sensitivity(true positive rate\recall for class 1)
#how many of the positive cases we catch?(formula=TP/TP+FN)
##precision-how relevent are the predictions?
#precisiom of class 0-how many of the negative predictios are correct?(formula=TN/TN+FN)
#precisiom of class 1-how many of the positive predictions were correct?(formula=TP/TP+FP)
#f1 score- harmonic mean of precision of racall and precision value (use for large no of classes in case of large no of classses )
##f1 score=2*(precision*recall)/(precision+recall)
##use mannual encooder in case od case sesitive data 
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%% tuning of model logistic regression /adjusting threshold
## store the predixted probabilities
Y_pred_prob=classifier.predict_proba(X_test)
print(Y_pred_prob)
#%%
Y_pred_class=[]
for value in Y_pred_prob[:,1]:
    if value > 0.58:
        Y_pred_class.append(1)
    else:
        Y_pred_class.append(0)
print(Y_pred_class)    
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred_class)
print(cfm)
print(classification_report(Y_test,Y_pred_class))
acc=accuracy_score(Y_test,Y_pred_class)
print('Accuracy of the model:',acc)
#%% to decide threshold
for a in np.arange(0.4,0.61,0.01):
    predict_mine = np.where(Y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :",
          cfm[1,0]," , type 1 error:", cfm[0,1]) 
#%% auc-0.5-0.6-poor model,0.6-0.7-badmodel,0.7-00.8-good model,0.8-0.9-vgood model,0.9-1.0-excellent model
from sklearn import metrics

fpr, tpr, z = metrics.roc_curve(Y_test, Y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)
#%%
 

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
#%% cross validation  k fold
#Using cross validation
#model evaluation
classifier=LogisticRegression()

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10,random_state=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#%% 
#model tuning
for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])


Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred
                   )
print('Accuracy of the model:',acc)
#%%% homework
adult_test=pd.read_csv(r'C:\Users\hp\Desktop\imarticus\python\datasets\adult_test.csv',header=None,delimiter=' *, *')
#%%
adult_test.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

adult_test.head()
#%%
adult_test=adult_test.drop(["education","fnlwgt"],axis=1)
adult_test.isnull().sum()
#%%
adult_test.dtypes
adult_test=adult_test.replace(['?'],np.nan)
adult_test.isnull().sum()
#%%
for value in ['workclass','occupation','native_country']:
    adult_test[value].fillna(adult_test[value].mode()[0],inplace=True)
adult_test.isnull().sum()
#%%
colname=[]
for x in adult_test.columns:
    if adult_test[x].dtype=='object':
        colname.append(x)
colname
#%%
from  sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname:
        adult_test[x]=le.fit_transform(adult_test[x])
adult_test.head()
#%%
X_test=adult_test.values[:,:-1]
Y_test=adult_test.values[:,-1]
#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_test)
X=scaler.transform(X_test)
print(X_test)
#%%
Y_test=Y_test.astype(int)
#%%
from sklearn.linear_model  import LogisticRegression 
#creat model
classifier=LogisticRegression()
#fitting training data to the model
classifier.fit(X_test,Y_test)
Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))
print(classifier.coef_)
print(classifier.intercept_)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%
plot_lm_1 = plt.figure(1)
plot_lm_1.set_figheight(8)
plot_lm_1.set_figwidth(12)

# fitted values (need a constant term for intercept)
model_fitted_y = lm.fittedvalues

plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'sales', data=new_df, lowess=True)

plot_lm_1.axes[0].set_title('Residuals vs Fitted')
plot_lm_1.axes[0].set_xlabel('Fitted values')
plot_lm_1.axes[0].set_ylabel('Residuals')