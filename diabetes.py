

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
