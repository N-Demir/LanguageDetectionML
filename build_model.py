import string
import json
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer

import numpy as np

def readInData(file_path, key):
	# Let's dejsonify
	strings = []
	with open(file_path, 'r') as f:
		for idx_line, line in enumerate(f.readlines()):
			strings.append(json.loads(line)[key])

	return strings

def trainKFoldModel(X, Y, model):
	accuracies = []

	print('Starting KFold')

	kf = KFold(n_splits=5, shuffle=True)
	for train_index, test_index in kf.split(X):
		train_X, train_Y = X[train_index], Y[train_index]
		valid_X, valid_Y = X[test_index], Y[test_index]

		# Word features
		vectorizer = CountVectorizer()
		vectorizer.fit(train_X)
		train_X = vectorizer.transform(train_X)
		valid_X = vectorizer.transform(valid_X)

		tfidf_transformator = TfidfTransformer()
		tfidf_transformator.fit(train_X)
		train_X = tfidf_transformator.transform(train_X)
		valid_X = tfidf_transformator.transform(valid_X)

		print('Fitting the data')
		model.fit(train_X, train_Y)

		print('Scoring')
		one_fold_accuracy = model.score(valid_X, valid_Y)
		print('Score on one fold = {}'.format(one_fold_accuracy))

		accuracies.append(one_fold_accuracy)

		print('Completed a kfold')

	return np.mean(accuracies)

def getFinalModel(X, Y, model):
	vectorizer.fit(train_X)
	train_X = vectorizer.transform(train_X).todense()
	valid_X = vectorizer.transform(valid_X).todense()
	train_X = normalize(train_X)
	valid_X = normalize(valid_X)

	# TODO: Some stuff

def main():

	X = np.array(readInData('train_X_languages_homework.json.txt', 'text'))
	Y = np.array(readInData('train_y_languages_homework.json.txt', 'classification'))

	# Use comments to choose a particular model to train
	# NB original accuracy = 0.7291591832430144
	# LR original accuracy = 0.6549672391650392 (10_000 0.5740603824620157)
	# SVM TODO: Change?

	clf = MultinomialNB()
	# clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
	# clf = SVC(kernel='linear')

	kfold_accuracy = trainKFoldModel(X, Y, clf)

	print('Overall KFold accuracy was {}'.format(kfold_accuracy))

	# Train on full dataset and save to disk
	# filename = 'finalized_model.sav'
	# final_model = getFinalModel(X, Y, clf)
	# pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
	main()
