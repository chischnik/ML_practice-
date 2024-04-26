#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Lese den Datensatz ein
wine_data = pd.read_csv('C:/Users/49152/Desktop/ML-Aufgabe/Aufgabe5 Korrelation/wine.data', header=None)

# Setze die Spaltennamen gemäß wine.info
columns = ['Winery', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 
           'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 
           'OD280/OD315 of diluted wines', 'Proline']
wine_data.columns = columns

# Berechne die Korrelation
correlation_matrix = wine_data.corr()

# Plot der Korrelationsmatrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korrelationsmatrix der Wine-Daten')
plt.show()

# Ausgabe der Korrelationen
print("Korrelationen der Attribute:")
print(correlation_matrix)


# In[ ]:




