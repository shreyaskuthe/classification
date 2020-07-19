import pandas as pd
import numpy as np
#%% loading testand train data
train_data=pd.read_csv(r'C:\Users\hp\Desktop\imarticus\python\datasets\risk_analytics_train.csv',header=0,index_col=0)
test_data=pd.read_csv(r'C:\Users\hp\Desktop\imarticus\python\datasets\risk_analytics_test.csv',header=0,index_col=0)
#%%preprocessing the training data
print(train_data.shape)
train_data.head()
#%% finding missing values
print(train_data.isnull().sum())
#print(train_data.shape)
#%% imputing categorical missing data with mode value
colname1=['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term']
for x in colname1:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True)
print(train_data.isnull().sum())
#%% imputing numerical mising data with mean values
train_data['LoanAmount'].fillna(round(train_data['LoanAmount'].mean(),0),
          inplace=True)
print(train_data.isnull().sum())
#%% imputing value for credit history column differentlty
train_data['Credit_History'].fillna(value=0,inplace=True)
#train_data['Credit_History']=train_data['Credit_History'].fillna(value=0)
print(train_data.isnull().sum())
#%% labdel encoding
colname=[]
for x in train_data.columns:
    if train_data[x].dtype=='object':
        colname.append(x)
colname
from  sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname:
        train_data[x]=le.fit_transform(train_data[x])
train_data.head()
#%% preprocessing the testing data set
#
#
#%% finding missing values
print(test_data.isnull().sum())
#print(test_data.shape)
#%% imputing categorical missing data with mode value
colname1=['Gender','Dependents','Self_Employed','Loan_Amount_Term']
for x in colname1:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True)
print(test_data.isnull().sum())
#%% imputing numerical mising data with mean values
test_data['LoanAmount'].fillna(round(train_data['LoanAmount'].mean(),0),
          inplace=True)
print(test_data.isnull().sum())
#%% imputing value for credit history column differentlty
test_data['Credit_History'].fillna(value=0,inplace=True)
#test_data['Credit_History']=test_data['Credit_History'].fillna(value=0)
print(test_data.isnull().sum())
#%% labdel encoding
colname=[]
for x in test_data.columns:
    if test_data[x].dtype=='object':
        colname.append(x)
colname
from  sklearn import preprocessing
le=preprocessing.LabelEncoder()
for x in colname:
        test_data[x]=le.fit_transform(test_data[x])
test_data.head()
#%% creating training and testing datasetsmamd running the model
X_train=train_data.values[:,:-1]
Y_train=train_data.values[:,-1]
Y_train=Y_train.astype(int)
#%% test_data.head
X_test=test_data.values[:,:]
#%%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#%% model building
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))
#%%
test_data=pd.read_csv(r'C:\Users\hp\Desktop\imarticus\python\datasets\risk_analytics_test.csv',header=0,index_col=0)
test_data['Y_predictions']=Y_pred
test_data.head()
#%%
test_data['Y_predictions']=test_data['Y_predictions'].map({1:'Eligible',0:'Not eligible'})
test_data.head()
#%%
test_data.to_csv('test_data.csv')
test_data.Y_predictions.value_counts()
#%% to check accuracy
#creat validation data
#Using cross validation
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

#classifier=svm.SVC(kernel='rbf',C=1.0,gamma=0.1) #75.89%
#classifier=KNeighborsClassifier(n_neighbors=11, metric='euclidean') #75.07%
classifier=svm.SVC(kernel='rbf',C=10.0,gamma=0.001) #77.03%
#classifier=LogisticRegression() #77.20%

#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())





















