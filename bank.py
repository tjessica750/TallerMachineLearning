
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

# 4. Crear una sección de métricas por cada modelo donde se de a conocer:
    # a. Accuracy de Entrenamiento (Train)
    # b. Accuracy de Validación
    # c. Accuracy de Test (Train)
def showMetrics(str_model,acc_train, acc_trainKN, acc_trainBC, acc_trainTC, acc_trainNB, acc_validation, acc_validationKN, acc_validationBC, acc_validationTC, acc_validationNB, acc_test, acc_testKN, acc_testBC, acc_testTC, acc_testNB):
   
   
    TableMetrics = pd.DataFrame({'Metric': ['LOGISTIC REGRESSION','KNEIGHBORNS','ADABOOST CLASSIFIER','DECISION TREE','GaussianNB'],
                     'Training Acurancy':[round(acc_train,4),round(acc_trainKN,4),round(acc_trainBC,4),round(acc_trainTC,4),round(acc_trainNB,4)],
                     'Validation Accurancy':[round(acc_validation,4),round(acc_validationKN,4),round(acc_validationBC,4),round(acc_validationTC,4),round(acc_validationNB,4)],
                     'Test Accurancy':[round(acc_test,4),round(acc_testKN,4),round(acc_testBC,4),round(acc_testTC,4),round(acc_testNB,4)]})
    print("punto 4")
    print(TableMetrics.sort_values) 
    
# 1. Entrenar 5 modelos con distintos algoritmos de Machine Learning

def metrics(str_model, y_test, y_pred):
    
    print('')



model = LogisticRegression()
model, acc_validation, acc_test, y_pred, acc_train =  metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_lg, AUC, fpr_lg, tpr_lg  = matrizConfusionAuc(model, x_test, y_test, y_pred)

matrizConfusion = confusion_matrix(y_test, y_pred)
nameModel = 'Logistic Regression'
Metrics = classification_report(y_test, y_pred)
metrics('Logistic Regression', y_test, y_pred)



model = KNeighborsClassifier(n_neighbors = 3)
model, acc_validationKN, acc_testKN, y_pred, acc_trainKN = metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_knn, AUCKN, fpr_knn, tpr_knn  = matrizConfusionAuc(model, x_test, y_test, y_pred)

matrizConfusionKN = confusion_matrix(y_test, y_pred)
nameModelKN = 'KNeighbors Classifier'
MetricsKN = classification_report(y_test, y_pred)
metrics('KNeighbors Classifier', y_test, y_pred)



model = AdaBoostClassifier(n_estimators=10)
model, acc_validationBC, acc_testBC, y_pred, acc_trainBC = metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_ada, AUCBC, fpr_ada, tpr_ada  = matrizConfusionAuc(model, x_test, y_test, y_pred)

matrizConfusionBC = confusion_matrix(y_test, y_pred)
nameModelBC = 'AdaBoost Classifier'
MetricsBC = classification_report(y_test, y_pred)
metrics('AdaBoost Classifier', y_test, y_pred)



model = DecisionTreeClassifier()
model, acc_validationTC, acc_testTC, y_pred, acc_trainTC = metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_dc, AUCTC, fpr_dc, tpr_dc  = matrizConfusionAuc(model, x_test, y_test, y_pred)
matrizConfusionTC = confusion_matrix(y_test, y_pred)
nameModelTC = 'Decision Tree Classifier'
MetricsTC = classification_report(y_test, y_pred)
metrics('Decision Tree Classifier', y_test, y_pred)



model = GaussianNB()
model, acc_validationNB, acc_testNB, y_pred, acc_trainNB = metricsTraining(model, x_train, x_test, y_train, y_test)
matriz_confusion_NB, AUCNB, fpr_NB, tpr_NB  = matrizConfusionAuc(model, x_test, y_test, y_pred)

matrizConfusionNB = confusion_matrix(y_test, y_pred)
nameModelNB = 'Gaussian Naive Bayes'
MetricsNB = classification_report(y_test, y_pred)
metrics('Gaussian Naive Bayes', y_test, y_pred)


showMetrics('KNeighborns', acc_train, acc_trainKN, acc_trainBC, acc_trainTC, acc_trainNB, acc_validation, acc_validationKN, acc_validationBC, acc_validationTC, acc_validationNB, acc_test, acc_testKN, acc_testBC, acc_testTC, acc_testNB) 





