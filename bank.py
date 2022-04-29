
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