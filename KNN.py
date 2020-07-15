#used for classificatrion and regrression or multi class classification 
##supervised machine learnong algorithm
#can perform upto min num oif observations
##KNN- if my neighbour is doinng this ill also do that 
##value of k is user defined 
#value of k start from usually 3 to 20
#this algorith is called as lazy algorithm
#works well upon noisy data
#to find value of k 
#sqrt of total nom of observation 
#try and test for various valus of k 
##k neighbourers Classifier ()
#k neighbourers Regressor ()
##Radious neighbors Classifier()
##Radious neighborsRegressor()
#not impcated by outliers
#%%
import pandas as pd
import numpy as np
training_data=pd.read_excel(r'C:\Users\hp\Desktop\imarticus\python\datasets\Titanic_Survival_Train.xls',index_col=0,header=0)
testing_data=pd.read_excel(r'C:\Users\hp\Desktop\imarticus\python\datasets\Titanic_Survival_Test.xls',index_col=0,header=0)
#%%
#%%
print(training_data)
training_data.info()
#%%
pd.set_option('display.max_columns',None)
print(training_data)
pd.set_option('display.max_rows',None)
print(training_data)
#%%
training_data.tail(10)
training_data.dtypes
training_data.info()
#%% featiure selection 
training_data=training_data[['Name','Pclass','Sex','Age','Survived']]
testing_data=testing_data[['Name','Pclass','Sex','Age','Survived']]
print(training_data.shape)
print(testing_data.shape)
#%% finding missing values
training_data.isnull().sum()
testing_data.isnull().sum()
#%% imputing numerical mising value with MEAN VALUE
training_data['Age'].fillna(int(training_data['Age'].mean()),inplace=True)
print(training_data.isnull().sum())

testing_data['Age'].fillna(int(testing_data['Age'].mean()),inplace=True)
print(testing_data.isnull().sum())
#%% converting categorical data to numeric
from  sklearn import preprocessing
colname=['Sex']

le=preprocessing.LabelEncoder()
for x in colname:
        training_data[x]=le.fit_transform( training_data[x])
        testing_data[x]=le.fit_transform( testing_data[x])
training_data.head()
testing_data.head()
#%%
X_train=training_data.values[:871,1:-1]
Y_train=training_data.values[:871,-1]
Y_train=Y_train.astype(int)

X_test=testing_data.values[:,1:-1]
Y_test=testing_data.values[:,-1]
Y_test=Y_test.astype(int)
#%%% scaler object should always be fitted only upon train data and same scale should used to transform training and testing data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#%% preictrion using KNeighbors_Classifier
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))),
                               metric='euclidean')
#fit the model on the data and predict the values
model_KNN.fit(X_train,Y_train)
Y_pred=model_KNN.predict(X_test)
print(list(zip(Y_test,Y_pred)))
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%
from sklearn.metrics import accuracy_score

for K in range(1,16):
    model_KNN = KNeighborsClassifier(K,metric="euclidean")
    model_KNN.fit(X_train, Y_train)
    Y_pred = model_KNN.predict(X_test)
    print ("Accuracy is ", accuracy_score(Y_test,Y_pred), "for K-Value:",K)
#%% eucledian,manhatten,minkowski
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=10,
                               metric='euclidean')
#fit the model on the data and predict the values
model_KNN.fit(X_train,Y_train)
Y_pred=model_KNN.predict(X_test)
print(list(zip(Y_test,Y_pred)))
#%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%
X_train=df1.values[:,:-1]
Y_train=df1.values[:,-1]
#%%


