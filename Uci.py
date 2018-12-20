#!/usr/bin/env python
#CLASIFICADOR DE ENFERMEDAD CARDIACA, CON BASE DE DATOS UCI-HEARTH, TOMADO DE KAGGLE.
#%%Librerias a utilizar y visualizacion de las variables
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


df=pd.read_csv('heart.csv')
y=df['target']
df=df.drop(['target'],axis=1)
plt.subplot(6,2,1);df['age'].plot(kind='hist',figsize=(15,15), rot=70, logx=False, logy=False, label='edad');plt.legend()
plt.subplot(6,2,2);df['cp'].plot(kind='hist  ', rot=70, logx=False, logy=False,label='tipo de dolor de pecho');plt.legend()
plt.subplot(6,2,3);df['ca'].plot(kind='hist', rot=70, logx=False, logy=False,label='numero de vasos mayores');plt.legend()
plt.subplot(6,2,4);df['trestbps'].plot(kind='hist', rot=70, logx=False, logy=False,label='presion sanguinea en reposo');plt.legend()
plt.subplot(6,2,5);df['chol'].plot(kind='hist', rot=70, logx=False, logy=False,label='colesterol mg/dl');plt.legend()
plt.subplot(6,2,6);df['restecg'].plot(kind='hist', rot=70, logx=False, logy=False,label='ecg en reposo');plt.legend()
plt.subplot(6,2,7);df['oldpeak'].plot(kind='hist', rot=70, logx=False, logy=False,label='Depresion ST');plt.legend()
plt.subplot(6,2,8);df['thalach'].plot(kind='hist', rot=70, logx=False, logy=False,label='Max Ritmo cardiaco');plt.legend()
plt.subplot(6,2,9);df['exang'].plot(kind='hist', rot=70, logx=False, logy=False,label='Angina por ejercicio');plt.legend()
plt.subplot(6,2,10);df['fbs'].plot(kind='hist', rot=70, logx=False, logy=False,label='Azucar en sangre mayor a 120');plt.legend()
plt.subplot(6,2,11);df['slope'].plot(kind='hist', rot=70, logx=False, logy=False,label='Pendiente del pico de ejercicio');plt.legend()
plt.subplot(6,2,12);df['thal'].plot(kind='hist', rot=70, logx=False, logy=False,label='tipo de defecto');plt.legend()
#In[]
df.head()
# In[]:
#Analisis exploratorio de relaciones de variables, se puede decir que la edad es una variable principal
plt.figure(figsize=(15,15))
plt.subplot(2,2,1,);plt.scatter(df['age'],df['trestbps'])
plt.subplot(2,2,2);plt.scatter(df['age'],df['chol'])
plt.subplot(2,2,3);plt.scatter(df['age'],df['thalach'])
plt.subplot(2,2,4);plt.scatter(df['age'],df['oldpeak'])

#In[]
#Hay 4 tipos de dolor de pecho y 3 tipos de ECG, 5 tipos de vasos mayores 3 tipos de slope, y 4 de thal, 
# se aplica getdummies.
cp=pd.get_dummies(df['cp'],dtype=float,prefix='cp')
ca=pd.get_dummies(df['ca'],dtype=float,prefix='ca')
ecg=pd.get_dummies(df['restecg'],dtype=float,prefix='ecg')
slope=pd.get_dummies(df['slope'],dtype=float,prefix='slope')
thal=pd.get_dummies(df['thal'],dtype=float,prefix='thal')


#Join para concatenar dataframes
df=df.drop(['cp','ca','restecg','slope','thal'],axis=1)
ds=df.join(cp)
ds=ds.join(ca)
ds=ds.join(ecg)
ds=ds.join(slope)
ds=ds.join(thal)
ds.describe() 

#Aplicare estandarizacion a las variables trestbps, chol,thalach y de ultimo la edad 
#%%
scaler = StandardScaler()
agestd=scaler.fit_transform(np.array(df['age']).reshape(-1,1))
treststd=scaler.fit_transform(np.array(df['trestbps']).reshape(-1,1))
cholstd=scaler.fit_transform(np.array(df['chol']).reshape(-1,1))
thalachstd=scaler.fit_transform(np.array(df['thalach']).reshape(-1,1))
ds['age']=agestd
ds['trestbps']=treststd
ds['chol']=cholstd
ds['thalach']=thalachstd
ds.describe()
X_train, X_test, y_train, y_test = train_test_split(ds.values, y, test_size=0.33)
clf=SVC()
param_grid = {"kernel": ['linear', 'poly', 'rbf','sigmoid'],'gamma': [0.01, 0.1, 0.3, 0.8, 1, 5,8,15,30],"C": [0.001, 0.01, 0.1, 1, 10,20,30]}

#%%Busqueda de hiperparametros con el metodo GridSearch
gs = GridSearchCV(clf, param_grid, scoring='accuracy', cv=3)
gs.fit(X_train, y_train)
gs.best_score_

#%%[Validacion con set de validacion]

y_pred_test=gs.predict(X_test)
z=accuracy_score(y_test,y_pred_test)
print(z)


#%% Los mejores parametros, precision del 88%
gs.best_params_

