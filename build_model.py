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
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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

		# tfidf_transformator = TfidfTransformer()
		# tfidf_transformator.fit(train_X)
		# train_X = tfidf_transformator.transform(train_X)
		# valid_X = tfidf_transformator.transform(valid_X)

		print('Fitting the data')
		model.fit(train_X, train_Y)

		print('Scoring')
		one_fold_accuracy = model.score(valid_X, valid_Y)
		print('Score on one fold = {}'.format(one_fold_accuracy))

		accuracies.append(one_fold_accuracy)

		print('Completed a kfold')

	return np.mean(accuracies)

def getFinalModel(X, Y, model):

	text_pipeline = Pipeline([('vectorizer', CountVectorizer()),
							  ('tfidf_transformator', TfidfTransformer()),
							  ('clf', model),
							 ])

	pipeline_parameters = {
		'vectorizer__analyzer': ['char_wb', 'char']
		'vectorizer__ngram_range': [(1, 6), (1, 9), (1, 20)],
		'tfidf_transformator__use_idf': [True],
		'clf__alpha': [1e-3],
	}

	grid_search_clf = GridSearchCV(text_pipeline, pipeline_parameters, verbose=5)
	return grid_search_clf.fit(X, Y)


def main():

	X = np.array(readInData('train_X_languages_homework.json.txt', 'text'))
	Y = np.array(readInData('train_y_languages_homework.json.txt', 'classification'))

	# Use comments to choose a particular model to train
	# NB Accuracies:
	# word = 0.7291591832430144
	# word + tfidf = 0.6218707534800128
	# GridSearch found best score as: 0.7695049883962963 with params {'clf__alpha': 0.001, 'tfidf_transformator__use_idf': False, 'vectorizer__ngram_range': (1, 2)}
	# character (4-grams) = 0.7537701959185161
	# character (6-grams) = 0.8363997416082494
	# character (2-8-grams) = 0.8408657718656044
	# GridSearch found best score as: 0.8417669529711064 with params {'clf__alpha': 0.001, 'tfidf_transformator__use_idf': True, 'vectorizer__ngram_range': (1, 6)}
	# LR Accuracies:
	# word = 0.6549672391650392
	# word (X.shape[0] = 10_000) = 0.5740603824620157
	# Linear SVM Accuracies:
	# word = 0.7006643124482357
	# word + tfidf = 0.7237122221016614
	# GridSearch found best score as: 0.7149396687162556 with params {'clf__alpha': 0.01, 'tfidf_transformator__use_idf': True, 'vectorizer__ngram_range': (1, 2)}
	# character (4-grams) = 0.7437297991604366
	# GridSearch found best score as: 0.7784689491271304 with params {'clf__alpha': 0.001, 'tfidf_transformator__use_idf': True, 'vectorizer__ngram_range': (1, 7)}

	clf = MultinomialNB()
	# clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
	# clf = SVC(kernel='linear')
	# clf = SGDClassifier(random_state=314159)

	# KFold training
	# kfold_accuracy = trainKFoldModel(X, Y, clf)
	# print('Overall KFold accuracy was {}'.format(kfold_accuracy))

	# Getting best model parameters
	gs = getFinalModel(X, Y, clf)
	print('GridSearch found best score as: {} with params {}'.format(gs.best_score_, gs.best_params_))

	# Train on full dataset and save to disk
	# filename = 'finalized_model.sav'
	# final_model = getFinalModel(X, Y, clf)
	# pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
	main()
