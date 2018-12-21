import string
import json
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
from joblib import dump

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

def getFinalModelParameters(X, Y, model):

	text_pipeline = Pipeline([('vectorizer', CountVectorizer()),
							  ('tfidf_transformator', TfidfTransformer()),
							  ('clf', model),
							 ])

	pipeline_parameters = {
		'vectorizer__analyzer': ['char_wb', 'word'],
		'vectorizer__ngram_range': [(1, 1), (1, 3), (1, 6)],
		'tfidf_transformator__use_idf': [False, True],
		'clf__alpha': [1e-3, 1e-2],
	}

	grid_search_clf = GridSearchCV(text_pipeline, pipeline_parameters, verbose=5)
	return grid_search_clf.fit(X, Y)

def writeModel(model):
	file_path = "finalized_model.joblib"
	dump(model, file_path) 

def main():

	X = np.array(readInData('train_X_languages_homework.json.txt', 'text'))
	Y = np.array(readInData('train_y_languages_homework.json.txt', 'classification'))

	## Use comments to choose a particular model to train
	clf = MultinomialNB()
	# clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
	# clf = SGDClassifier(random_state=314159)

	## KFold training
	# kfold_accuracy = trainKFoldModel(X, Y, clf)
	# print('Overall KFold accuracy was {}'.format(kfold_accuracy))

	## Getting best model parameters
	gs_clf = getFinalModelParameters(X, Y, clf)
	print('GridSearch found best score as: {} with params {}'.format(gs_clf.best_score_, gs_clf.best_params_))

	## Save performance expectation
	open('performance.txt', 'w+').write('GridSearch found best score as: {} with params {}'.format(gs_clf.best_score_, gs_clf.best_params_))

	## Save model
	final_model = gs_clf.fit(X, Y)
	writeModel(final_model)


if __name__ == "__main__":
	main()
