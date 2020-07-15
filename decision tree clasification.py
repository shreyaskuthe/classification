import pandas as pd 
import numpy as np
#%%%
cars_data=pd.read_csv(r'C:\Users\hp\Desktop\imarticus\python\datasets\cars.csv',header=None)
cars_data.head()
#%%%
# assaining column headers
cars_data.columns=['buying','maint','doors','persons','lug_boot','safety','classes']
#%%%
cars_data.head()
#%%%
#checking missing values
cars_data.isnull().sum()
#%%%
#creating a copy of the data frame
cars_df=pd.DataFrame.copy(cars_data)
#%%%
colname=cars_df.columns
colname
#%%%

from sklearn import preprocessing
le=preprocessing.LabelEncoder()

for x in colname:
    cars_df[x]=le.fit_transform(cars_df[x])
#%%%
cars_df.head
#%%%
cars_df.classes.value_counts()

#acc==>0
#good==>1
#unacc==>2
#vgood==>3
#%%%
X=cars_df.values[:,:-1]
Y=cars_df.values[:,-1]
Y=Y.astype(int)
#%%%
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X)

X=scaler.transform(X)
#%%%
from sklearn.model_selection import train_test_split

#Split the data into test and train

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)
#%%%
from sklearn.tree import DecisionTreeClassifier
#%%%
#create the model object

model_DT=DecisionTreeClassifier(criterion='gini',random_state=10)

#fit the model on the data and predict the values
model_DT.fit(X_train,Y_train)
#%%%
#predit using the model
Y_pred=model_DT.predict(X_test)
#print (Y_pred)
#print(list(zip(Y_test,Y_pred)))
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%
print(list(zip(colname,model_DT.feature_importances_)))
#%%%
from sklearn import tree
with open("model_DecisionTree.txt","w") as f:

        f = tree.export_graphviz(model_DT, feature_names=colname[:-1],
                         out_file=f)

#generate the file and upload the code in webgraphviz.com to plot the decision tree
#%%%
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=50,gamma=0.1)
svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%
#Using cross validation
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

#classifier=svm.SVC(kernel='rbf',C=80,gamma=0.1) #98.75
#classifier=KNeighborsClassifier(n_neighbors=11, metric='euclidean') #87.50
#classifier=svm.SVC(kernel='rbf',C=70.0,gamma=0.1) #98.67
#classifier=LogisticRegression() #68.64

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
#%%%%
#predicting using the bagging classifier
from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier(100,random_state=30)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%
#predicting using the Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=101,random_state=10)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
model=AdaBoostClassifier(n_estimators=40,base_estimator=DecisionTreeClassifier(),random_state=10)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
model=GradientBoostingClassifier(n_estimators=250,random_state=10)
#fit the model on the data and predict the values
model=model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# create the sub models
estimators = []
#model1 = LogisticRegression()
#estimators.append(('log', model1))
model2 = DecisionTreeClassifier(criterion='gini',random_state=10)
estimators.append(('cart', model2))
model3 = SVC(kernel="rbf", C=50,gamma=0.1)
estimators.append(('svm', model3))
#model4 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
#estimators.append(('knn', model4))


# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#print(Y_pred)
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%