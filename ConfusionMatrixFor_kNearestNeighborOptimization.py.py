#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[31]:


# Laden der Abalone-Daten
abalone_data = pd.read_csv("C:/Users/49152/Desktop/Aufgabe15 Confusion metrix/abalone.data", header=None)
abalone_data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']


# In[32]:


# Laden der Iris-Daten
iris_data = pd.read_csv("C:/Users/49152/Desktop/Aufgabe15 Confusion metrix/iris.data", header=None)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']


# In[33]:


# Funktion zur Erstellung der Confusion Matrix und Bestimmung des besten K
def confusion_matrix_for_knn(dataset, attributes, target_column):
    X = dataset[attributes]
    y = dataset[target_column]
    
    # Normalisierung der Daten
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Aufteilung in Trainings- und Testdatensatz
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)
    
    best_accuracy = 0
    best_k = 0
    cm_best = None
    
    # Iteration über verschiedene Werte von k
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        accuracy = np.mean(y_pred == y_test)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            cm_best = confusion_matrix(y_test, y_pred)
    
    # Visualisierung der Confusion Matrix
    plt.imshow(cm_best, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(dataset[target_column].unique()))
    plt.xticks(tick_marks, dataset[target_column].unique(), rotation=45)
    plt.yticks(tick_marks, dataset[target_column].unique())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    print("Bestes k: ", best_k)
    print("Genauigkeit: ", best_accuracy)

# Anwendung der Funktion auf den Abalone-Datensatz
confusion_matrix_for_knn(abalone_data, ['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight'], 'Sex')

# Anwendung der Funktion auf den Iris-Datensatz
confusion_matrix_for_knn(iris_data, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 'class')


# In[ ]:




