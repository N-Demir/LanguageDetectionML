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
import numpy as np
from joblib import load

def readInData(file_path, key):
	# Let's dejsonify
	strings = []
	with open(file_path, 'r') as f:
		for idx_line, line in enumerate(f.readlines()):
			strings.append(json.loads(line)[key])

	return strings

def outputData(file_path, datas):
	with open(file_path, 'w+') as f:
		for data in datas:
			f.write('{}\n'.format(data))

def main():
	X = np.array(readInData('test_X_languages_homework.json.txt', 'text'))

	file_path = "finalized_model.joblib"
	trained_clf = load(file_path) 

	predictions = trained_clf.predict(X)

	outputData('predictions.txt', predictions)

if __name__ == "__main__":
	main()