import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics 

sns.set(style="darkgrid")
data = pd.read_csv("mushrooms.csv")
 
encoder = LabelEncoder()

for col in data.columns:
	data[col] = encoder.fit_transform(data[col])

X = data.iloc[:,1:23]
y = data.iloc[:,0]

X = StandardScaler().fit_transform(X)

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

log = LogisticRegression()

log.fit(X_train, y_train)

y_pred = log.predict(X_test)

conf = metrics.confusion_matrix(y_test, y_pred) 

print(cross_val_score(log, X, y, cv=10, scoring = 'accuracy').mean())
print(conf)

print('\n \n \n \n')

vector = SVC()
vector.fit(X_train, y_train)

y_pred_vector = vector.predict(X_test)

conf_vector = metrics.confusion_matrix(y_test, y_pred_vector) 

print(cross_val_score(vector, X, y, cv=10, scoring = 'accuracy').mean())
print(conf_vector)
