import string
import json
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

import numpy as np

kf = KFold(n_splits=5, shuffle=True)

def readInData(file_path, key):
	# Let's dejsonify
	strings = []
	with open(file_path, 'r') as f:
		for idx_line, line in enumerate(f.readlines()):
			dejsonified = json.loads(line)
			
			s = dejsonified[key]

			exclude = set(string.punctuation)
			s = ' '.join(ch for ch in s if ch not in exclude and ch != ' ')

			strings.append(s)

	return strings

def getCharacterFeatures(X):
	characters = set()

	for line in X:
		chs = line.split()
		for ch in chs:
			characters.add(ch)

	return list(characters)

def transformToFeatures(X, feature_list):
	features = []

	for line in X:
		line_as_feature = [0 for _ in range(len(feature_list))]

		count = Counter(line)

		for idx_feature, feature in enumerate(feature_list):
			line_as_feature[idx_feature] = count[feature]

		features.append(line_as_feature)

	return np.array(features)

X = np.array(readInData('train_X_languages_homework.json.txt', 'text'))
Y = np.array(readInData('train_y_languages_homework.json.txt', 'classification'))


def trainModel(X, Y, model):
	accuracies = []

	print('Starting KFold')

	for train_index, test_index in kf.split(X):
		train_X, train_Y = X[train_index], Y[train_index]
		valid_X, valid_Y = X[test_index], Y[test_index]

		# Character features
		feature_list = getCharacterFeatures(train_X)
		train_X = transformToFeatures(train_X, feature_list)
		valid_X = transformToFeatures(valid_X, feature_list)


		# Word features with countvec
		# vectorizer.fit(train_X)
		# train_X = vectorizer.transform(train_X).todense()
		# valid_X = vectorizer.transform(valid_X).todense()
		# train_X = normalize(train_X)
		# valid_X = normalize(valid_X)

		print('Fitting the data')
		model.fit(train_X, train_Y)

		print('Scoring')
		one_fold_accuracy = model.score(valid_X, valid_Y)
		print('Score on one fold = {}'.format(one_fold_accuracy))

		accuracies.append(one_fold_accuracy)

		print('Completed a kfold')

	return np.mean(accuracies)

# NB
# clf = MultinomialNB() # 0.7291591832430144
clf = MultinomialNB()

# LR
# clf = LogisticRegression(solver='lbfgs', multi_class='multinomial') # 0.6549672391650392
# With 10_000 - 0.5740603824620157

# SVM
# clf = SVC(kernel='linear')

print(trainModel(X, Y, clf))

# Shift to try to use just character based features




