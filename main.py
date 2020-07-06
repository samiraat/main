

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cPickle
import gzip

from sklearn.model_selection import cross_val_predict
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

#############Iris
df = pd.read_csv('dataset/iris.data', header=0)
df['label'] = df['label'].map( {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2} ).astype(int)

data = df.values

X = data[0::, 0:-1:]
y = data[0::, -1]


#SVM
clf = SVC(probability=True)
predicted = cross_val_predict(clf, X, y, cv=5)
iris_svm_score = metrics.accuracy_score(y, predicted)

#KNN
clf = KNeighborsClassifier(3)
predicted = cross_val_predict(clf, X, y, cv=5)
iris_knn_score = metrics.accuracy_score(y, predicted)

#GaussianNB
clf = NaiveBayes()
predicted = cross_val_predict(clf, X, y, cv=5)
iris_NB_score = metrics.accuracy_score(y, predicted)

#RandomForest
clf = RandomForestClassifier()
predicted = cross_val_predict(clf, X, y, cv=5)
iris_RF_score = metrics.accuracy_score(y, predicted)

#LR
clf = LogisticRegression()
predicted = cross_val_predict(clf, X, y, cv=5)
iris_LR_score = metrics.accuracy_score(y, predicted)

#LDA
clf = LinearDiscriminantAnalysis()
predicted = cross_val_predict(clf, X, y, cv=5)
iris_LDA_score = metrics.accuracy_score(y, predicted)

#QDA
clf = QuadraticDiscriminantAnalysis()
predicted = cross_val_predict(clf, X, y, cv=5)
iris_QDA_score = metrics.accuracy_score(y, predicted)


###### PLOT IRIS

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

names = ["SVM", "KNN", "NaiveBayes", "RF", "LR", "LDA", "QDA"]
iris_scores = [iris_svm_score, iris_knn_score, iris_NB_score, iris_RF_score, iris_LR_score, iris_LDA_score, iris_QDA_score]

for i in range(len(names)):
	log_entry = pd.DataFrame([[names[i], iris_scores[i]]], columns=log_cols)
	log = log.append(log_entry)

sns.set_color_codes("muted")
sns.pointplot(y='Accuracy', x='Classifier', data=log, color="b")

plt.xlabel('Accuracy')
plt.title('Iris Classifier Accuracy')
plt.show()


#############Wine
df = pd.read_csv('dataset/wine.data', header=0)

data = df.values

X = data[0::, 1::]
y = data[0::, 0]

#SVM
clf = SVC(probability=True)
predicted = cross_val_predict(clf, X, y, cv=5)
wine_svm_score = metrics.accuracy_score(y, predicted)

#KNN
clf = KNeighborsClassifier(3)
predicted = cross_val_predict(clf, X, y, cv=5)
wine_knn_score = metrics.accuracy_score(y, predicted)

#GaussianNB
clf = NaiveBayes()
predicted = cross_val_predict(clf, X, y, cv=5)
wine_NB_score = metrics.accuracy_score(y, predicted)

#RandomForest
clf = RandomForestClassifier()
predicted = cross_val_predict(clf, X, y, cv=5)
wine_RF_score = metrics.accuracy_score(y, predicted)

#LR
clf = LogisticRegression()
predicted = cross_val_predict(clf, X, y, cv=5)
wine_LR_score = metrics.accuracy_score(y, predicted)

#LDA
clf = LinearDiscriminantAnalysis()
predicted = cross_val_predict(clf, X, y, cv=5)
wine_LDA_score = metrics.accuracy_score(y, predicted)

#QDA
clf = QuadraticDiscriminantAnalysis()
predicted = cross_val_predict(clf, X, y, cv=5)
wine_QDA_score = metrics.accuracy_score(y, predicted)

###### PLOT WINE

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

names = ["SVM", "KNN", "NaiveBayes", "RF", "LR", "LDA", "QDA"]
wine_scores = [wine_svm_score, wine_knn_score, wine_NB_score, wine_RF_score, wine_LR_score, wine_LDA_score, wine_QDA_score]

for i in range(len(names)):
	log_entry = pd.DataFrame([[names[i], wine_scores[i]]], columns=log_cols)
	log = log.append(log_entry)

sns.set_color_codes("muted")
sns.pointplot(y='Accuracy', x='Classifier', data=log, color="b")

plt.xlabel('Accuracy')
plt.title('Wine Classifier Accuracy')
plt.show()


#############Mnist
def load_mnist_data():
	f = gzip.open('dataset/mnist.pkl.gz', 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()
	return training_data

data = load_mnist_data()

X = data[0]
y = data[1]

#SVM
clf = SVC(probability=True)
predicted = cross_val_predict(clf, X, y, cv=3)
mnist_svm_score = metrics.accuracy_score(y, predicted)


#KNN
clf = KNeighborsClassifier(3)
predicted = cross_val_predict(clf, X, y, cv=3)
mnist_knn_score = metrics.accuracy_score(y, predicted)


#GaussianNB
clf = NaiveBayes()
predicted = cross_val_predict(clf, X, y, cv=3)
mnist_NB_score = metrics.accuracy_score(y, predicted)


#RandomForest
clf = RandomForestClassifier()
predicted = cross_val_predict(clf, X, y, cv=3)
mnist_RF_score = metrics.accuracy_score(y, predicted)


#LR
clf = LogisticRegression()
predicted = cross_val_predict(clf, X, y, cv=3)
mnist_LR_score = metrics.accuracy_score(y, predicted)



#LDA
clf = LinearDiscriminantAnalysis()
predicted = cross_val_predict(clf, X, y, cv=3)
mnist_LDA_score = metrics.accuracy_score(y, predicted)


#QDA
clf = QuadraticDiscriminantAnalysis()
predicted = cross_val_predict(clf, X, y, cv=3)
mnist_QDA_score = metrics.accuracy_score(y, predicted)


###### PLOT MNIST

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

names = ["SVM", "KNN", "NaiveBayes", "RF", "LR", "LDA", "QDA"]
mnist_scores = [mnist_svm_score, mnist_knn_score, mnist_NB_score, mnist_RF_score, mnist_LR_score, mnist_LDA_score, mnist_QDA_score]

for i in range(len(names)):
	log_entry = pd.DataFrame([[names[i], mnist_scores[i]]], columns=log_cols)
	log = log.append(log_entry)

sns.set_color_codes("muted")
sns.pointplot(y='Accuracy', x='Classifier', data=log, color="b")

plt.xlabel('Accuracy')
plt.title('Mnist Classifier Accuracy')
plt.show()