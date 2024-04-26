import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

path = "C:/Users/oumar/OneDrive/Documents/Studium/artest.csv"
artest = np.genfromtxt(path, delimiter='', dtype='str')
print(artest)
artest_int = []
for n in artest:
    map_obj = map(int, str(n))
    artest_int.append(list(map_obj))
df_artest = pd.DataFrame(
    data=np.array(artest_int, dtype=int),
    columns=['product_%d' % i for i in range(1, 51)]
)


support_products = apriori(df_artest, min_support=0.01, use_colnames=True)
print(support_products)
res= association_rules(support_products, metric='confidence', min_threshold=0.1)
print(res)
