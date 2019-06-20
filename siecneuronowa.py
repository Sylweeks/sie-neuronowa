import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
# importing the dataset
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split, cross_val_score
data = load_breast_cancer()
accuracy_all = []
cvs_all = []
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print("zlosliwy i nie\n",label_names)
# 0-zlosliwy 1-lagodny
print("wyniki\n",labels)
#zbior cech guza
print(feature_names)
#30 atrybutow dla 569 przypadkow
print(features)
# PODZIAL DANYCH NA 33% TESTOWYCH, RESZTA TRENINGOWA
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

from sklearn.neural_network import MLPClassifier
start = time.time()
clf = MLPClassifier(hidden_layer_sizes=(100,100,100),activation='tanh',
                     solver='sgd', verbose=10,random_state=21,tol=0.000000001)
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
print("MLPClassifier: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))






# initializing the classifier
gnb = GaussianNB()
# szkolenie modelu
model = gnb.fit(train, train_labels)
# wyniki dla zestawu testowego
predictions = gnb.predict(test)
# printing the predictions
print(predictions)
# evaluating the accuracy /sprawdzanie wynikow testów i dokladnosci
print(accuracy_score(test_labels, predictions))
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
print("\n\n\nNaive bayes")
start = time.time()
clf = GaussianNB()
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
#accuracy_all.append(accuracy_score(prediction, y_test))
print("Accuracy: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))

## stochastyczy spadek gradientu jest bardzo losoyw            =============================================
from sklearn.linear_model import SGDClassifier
start = time.time()
clf = SGDClassifier()
clf.fit(train, train_labels)
prediction = clf.predict(test)
#scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
#accuracy_all.append(accuracy_score(prediction, y_test))
#cvs_all.append(np.mean(scores))
print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))

####SVC masszyna wektorpw nosniych############################
from sklearn.svm import SVC, NuSVC, LinearSVC

start = time.time()
clf = SVC()
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))

#Nu SVC+++++++++++++++++++++++++++++++++++
start = time.time()
clf = NuSVC()
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))

##Linear SVC=========================================
start = time.time()
clf = LinearSVC()
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))

#Najblizszy sąsiad==============================
from sklearn.neighbors import KNeighborsClassifier
start = time.time()
clf = KNeighborsClassifier()
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
print("Nearest Neighbors: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))

####metody drzewa i lasy
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
#Random Forest=========================
start = time.time()
clf = RandomForestClassifier()
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
print("Random Forest: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))
#######dodatkowe drzewa========================
start = time.time()
clf = ExtraTreesClassifier()
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
print("Extra Trees: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))

"""
"""####drzewo decyzyjne------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
start = time.time()
clf = DecisionTreeClassifier()
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
print("Drzewo decyzyjne: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))

#============clasyfikator sklearn neural_network mlcpc"""
from sklearn.neural_network import MLPClassifier
start = time.time()
clf = MLPClassifier(hidden_layer_sizes=(100,100,100),activation='tanh',
                     solver='sgd', verbose=10,random_state=21,tol=0.000000001)
clf.fit(train, train_labels)
prediction = clf.predict(test)
end = time.time()
print("MLPClassifier: {0:.2%}".format(accuracy_score(prediction, test_labels)))
print("Execution time: {0:.5} seconds \n".format(end-start))
