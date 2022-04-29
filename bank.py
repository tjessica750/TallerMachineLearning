
"""
Created on Thu Apr 28 

@author: Jessica Torres

"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold


#clasificadores
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import roc_auc_score


# Quitar los waring
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


url = 'bank-full.csv'
data = pd.read_csv(url)



# 2. Limpiar y normalizar la data


rank = [16, 32, 48, 64, 80, 96]
Name = ['1', '2', '3', '4', '5' ]
data.age = pd.cut(data.age, rank, labels=Name)
data.marital.replace(['single','married','divorced'],[0,1,2], inplace=True)
data.job.replace(['management', 'technician', 'entrepreneur', 'blue-collar', 'unknown','retired',
 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid','student']
 ,[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
data.education.replace(['unknown','primary','secondary','tertiary'],[0,1,2,3], inplace=True)
data.default.replace(['no','yes'],[0, 1], inplace=True)
data.housing.replace(['no','yes'],[0, 1], inplace=True)
data.loan.replace(['no','yes'],[0, 1], inplace=True)
data.drop(['contact'], axis=1, inplace = True ) 
data.month.replace(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
 ,[1,2,3,4,5,6,7,8,9,10,11,12],inplace = True)
data.poutcome.replace(['unknown','failure','success','other'],[0, 1, 2, 3], inplace=True) 
data.y.replace(['no','yes'],[0, 1], inplace=True)
# data limpia

# 3. Dividir la data en train y test
# creacion del modelo
x = np.array(data.drop(['y'],1))
y = np.array(data.y)
# 0 is not, 1 is yes
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

def metricsTraining(model, x_train, x_test, y_train, y_test):
    kfold = KFold(n_splits=10)
    cvscores = [] 
    for train, test in kfold.split(x_train, y_train):
        model.fit(x_train[train], y_train[train])
        scores = model.score(x_train[test], y_train[test])
        cvscores.append(scores)
    y_pred = model.predict(x_test)
    accuracy_train = accuracy_score(model.predict(x_train), y_train)
    accuracy_validation = np.mean(cvscores)
    accuracy_test = accuracy_score(y_pred, y_test)
    return model, accuracy_validation, accuracy_test, y_pred, accuracy_train

def matrizConfusionAuc(model, x_test, y_test, y_pred):
    matriz_confusion = confusion_matrix(y_test, y_pred)
    probs = model.predict_proba(x_test)
    probs = probs[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    AUC = roc_auc_score(y_test, probs)
    return matriz_confusion, AUC, fpr, tpr


