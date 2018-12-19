import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import numpy as np

kf = KFold(n_splits=5, shuffle=True)
vectorizer = CountVectorizer()

def readInData(file_path, key):
	# Let's dejsonify
	strings = []
	with open(file_path, 'r') as f:
		for line in f.readlines():
			dejsonified = json.loads(line)
			strings.append(dejsonified[key])
	return strings

X = readInData('train_X_languages_homework.json.txt', 'text')
Y = readInData('train_y_languages_homework.json.txt', 'classification')

def trainNaiveBayes(X, Y):
	accuracies = []

	for train_index, test_index in kf.split(X):
		train_X, train_Y = X[train_index], Y[train_index]
		valid_X, valid_Y = X[test_index], Y[test_index]

		vectorizer.fit(train_X)
		train_X = vectorizer.transform(train_X)
		valid_X = vectorizer.transform(valid_X)

		clf = MultinomialNB()
		clf.fit(train_X, train_Y)

		accuracies.append(clf.score(valid_X, valid_Y))

		print('Completed a kfold')

	return np.mean(accuracies)

def trainLogisticRegression(X, Y):
	accuracies = []

	for train_index, test_index in kf.split(X):
		train_X, train_Y = X[train_index], Y[train_index]
		valid_X, valid_Y = X[test_index], Y[test_index]

		vectorizer.fit(train_X)
		train_X = vectorizer.transform(train_X)
		valid_X = vectorizer.transform(valid_X)

		clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
		clf.fit(train_X, train_Y)

		accuracies.append(clf.score(valid_X, valid_Y))

		print('Completed a kfold')

	return np.mean(accuracies)

# print(trainNaiveBayes(np.array(X), np.array(Y))) # 0.7291591832430144

print(trainLogisticRegression(np.array(X), np.array(Y)))





