

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

url = 'diabetes.csv'
data = pd.read_csv(url)



# 2. Limpiar y normalizar la data

rankGlucose = [-1, 39, 79, 119, 159, 199]
NameGlucose = ['1', '2', '3', '4','5']
data.Glucose = pd.cut(data.Glucose, rankGlucose, labels=NameGlucose)
rankDiabetesPedigreeFunction = [-1, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
NameDiabetesPedigreeFunction = ['1', '2', '3', '4','5','6','7','8','9','10','11']
data.DiabetesPedigreeFunction = pd.cut(data.DiabetesPedigreeFunction, rankDiabetesPedigreeFunction, labels=NameDiabetesPedigreeFunction)
rankAge = [20, 24, 29, 41, 81]
NameAge = ["1", "2", "3", "4"]
data.Age = pd.cut(data.Age, rankAge, labels=NameAge)

# 3. Dividir la data en train y test
# creacion del modelo


x = np.array(data.drop(['Outcome'],1))
y = np.array(data.Outcome)
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

