import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVC

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

X = np.array(readInData('train_X_languages_homework.json.txt', 'text'))
Y = np.array(readInData('train_y_languages_homework.json.txt', 'classification'))

def trainModel(X, Y, model):
	accuracies = []

	print('Starting KFold')

	for train_index, test_index in kf.split(X):
		train_X, train_Y = X[train_index], Y[train_index]
		valid_X, valid_Y = X[test_index], Y[test_index]

		vectorizer.fit(train_X)
		train_X = vectorizer.transform(train_X)
		valid_X = vectorizer.transform(valid_X)

		print('Fitting the data')
		model.fit(train_X, train_Y)

		print('Scoring')
		accuracies.append(model.score(valid_X, valid_Y))

		print('Completed a kfold')

	return np.mean(accuracies)

# NB
# clf = MultinomialNB() # 0.7291591832430144

# LR
# clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')

# SVM
clf = SVC(kernel='linear')

print(trainModel(X, Y, clf))





