#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[10]:


# Load the dataset
data = pd.read_csv("C:/Users/49152/Desktop/Aufgabe07klassifikationsanalyse/cl_problem.data",  sep='\s+')



# In[11]:


data 


# In[12]:


# Split the dataset into two parts: a-b class and c-d class
data_ab = data.iloc[:200]
data_cd = data.iloc[200:]

def evaluate_classifier(X_train, X_test, y_train, y_test, classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# In[13]:


# Task (a): Vergleich der Performanz von Entscheidungsbaum und Naive Bayes für den gesamten Datensatz
X = data.drop(columns=['classattr'])
y = data['classattr']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier()
nb_classifier = GaussianNB()

dt_accuracy = evaluate_classifier(X_train, X_test, y_train, y_test, dt_classifier)
nb_accuracy = evaluate_classifier(X_train, X_test, y_train, y_test, nb_classifier)

print("Decision Tree accuracy (whole dataset):", dt_accuracy)
print("Naive Bayes accuracy (whole dataset):", nb_accuracy)


# In[14]:


# Task (b): Vergleich der Performanz von Entscheidungsbaum und Naive Bayes für Klassen a und b
X_ab = data_ab.drop(columns=['classattr'])
y_ab = data_ab['classattr']
X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(X_ab, y_ab, test_size=0.2, random_state=42)

dt_accuracy_ab = evaluate_classifier(X_train_ab, X_test_ab, y_train_ab, y_test_ab, dt_classifier)
nb_accuracy_ab = evaluate_classifier(X_train_ab, X_test_ab, y_train_ab, y_test_ab, nb_classifier)

print("\nDecision Tree accuracy (classes a and b):", dt_accuracy_ab)
print("Naive Bayes accuracy (classes a and b):", nb_accuracy_ab)


# In[15]:


# Task (c): Vergleich der Performanz von Entscheidungsbaum und Naive Bayes für Klassen c und d
X_cd = data_cd.drop(columns=['classattr'])
y_cd = data_cd['classattr']
X_train_cd, X_test_cd, y_train_cd, y_test_cd = train_test_split(X_cd, y_cd, test_size=0.2, random_state=42)

dt_accuracy_cd = evaluate_classifier(X_train_cd, X_test_cd, y_train_cd, y_test_cd, dt_classifier)
nb_accuracy_cd = evaluate_classifier(X_train_cd, X_test_cd, y_train_cd, y_test_cd, nb_classifier)

print("\nDecision Tree accuracy (classes c and d):", dt_accuracy_cd)
print("Naive Bayes accuracy (classes c and d):", nb_accuracy_cd)


# In[16]:


# Task (d): Vergleich der Performanz von Entscheidungsbaum und Naive Bayes für Klassen a und b mit attr1 und attr4
X_ab = data_ab[['attr1', 'attr4']]  # Nur attr1 und attr4 verwenden
y_ab = data_ab['classattr']
X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(X_ab, y_ab, test_size=0.2, random_state=42)

dt_classifier_ab = DecisionTreeClassifier()
nb_classifier_ab = GaussianNB()

dt_accuracy_ab = evaluate_classifier(X_train_ab, X_test_ab, y_train_ab, y_test_ab, dt_classifier_ab)
nb_accuracy_ab = evaluate_classifier(X_train_ab, X_test_ab, y_train_ab, y_test_ab, nb_classifier_ab)

print("Decision Tree accuracy for classes a and b using attr1 and attr4:", dt_accuracy_ab)
print("Naive Bayes accuracy for classes a and b using attr1 and attr4:", nb_accuracy_ab)


# In[17]:


# Visualize the performance comparison
labels = ['Decision Tree', 'Naive Bayes']
accuracies = [dt_accuracy_ab, nb_accuracy_ab]

plt.bar(labels, accuracies, color=['blue', 'green'])
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Performance Comparison (Classes a and b)')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
plt.show()


# In[ ]:




