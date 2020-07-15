import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
#%%%
df=datasets.load_iris()
df.data #independent variables
df.target #dependent variables
df.feature_names #column header
print(df.DESCR) #dataset desciption
#%%%
X=df.data
Y=df.target
#%%%
X.shape
#%%%
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X)

X=scaler.transform(X)
#%%%
from sklearn.model_selection import train_test_split

#Split the data into test and train

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

#%%%
#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_
print(explained_variance)
#%%%
#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_
print(explained_variance)

#%%%
#Fitting SVC to Training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',gamma=0.1,C=1)
classifier.fit(X_train,Y_train)
#Predicting thr test set results
Y_pred=classifier.predict(X_test)
#%%%
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
#%%%
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.5, cmap = ListedColormap(('pink', 'yellow', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'black', 'red'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
#%%%
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
alpha = 0.5, cmap = ListedColormap(('pink', 'orange', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
#%%%