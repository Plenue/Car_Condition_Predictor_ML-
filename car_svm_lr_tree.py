import numpy as numpy
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("cardata.csv", names=["buying","maint","doors","persons","lug_boot","safety","class"])

clean={'class': {'unacc':4, 'acc':3,'good':2,'vgood':1}}
data.replace(clean,inplace = True)
y=data['class']
data.drop(['class'],axis=1,inplace = True)

data=pd.get_dummies(data)
print(data)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(data,y,random_state=0)

from sklearn.preprocessing import StandardScaler
fn=StandardScaler()
fn.fit(xtrain)
xtrain_std=fn.transform(xtrain)
xtest_std=fn.transform(xtest)


from sklearn import svm
from sklearn.svm import SVC
svc=svm.SVC(kernel='linear',C=1)
svc.fit(xtrain_std,ytrain)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)
model.score(xtrain_std,ytrain)
print(model.coef_.shape)
print(model.intercept_.shape)

from sklearn import tree
tmodel=tree.DecisionTreeClassifier(criterion='gini')
tmodel.fit(xtrain,ytrain)
tmodel.score(xtrain,ytrain)


from sklearn.metrics import accuracy_score 
svmpre=svc.predict(xtest_std)
print('SVM')
print('Misclassified samples: %d' %(ytest!=svmpre).sum())
print('Classification Accuracy: %.2f ' %accuracy_score(ytest,svmpre))
lrpre= model.predict(xtest)
print('Logistic Regression')
print('Misclassified samples: %d' %(ytest!=lrpre).sum())
print('Classification Accuracy: %.2f ' %accuracy_score(ytest,lrpre))
tpre= tmodel.predict(xtest)
print('Decision trees')
print('Misclassified samples: %d' %(ytest!=tpre).sum())
print('Classification Accuracy: %.2f ' %accuracy_score(ytest,tpre))