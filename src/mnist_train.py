import os, argparse, joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import model_dispatcher, config

def train_test_model(model_name, cv):
	'''
	Prints the mean cross validation score for the specified model on MNIST training set.
	Trains the model on MNIST train set and prints accuracy score w.r.t. the test set.
	Saves the trained model to the OUTPUT_PATH in config.

	Parameters: 
		model_name: str
			one of {'decision_tree', 'rf', 'log_reg'}
		cv: int
			number of cross validation folds
				
	Returns: 
		None
	'''

	#read train and test files
	train = pd.read_csv(config.TRAIN_FILE)
	test = pd.read_csv(config.TEST_FILE)
	
	x_train = train
	x_test = test
	y_train = x_train.pop('label')
	y_test = x_test.pop('label')
	
	#retrieve the model object from model_dispatcher
	
	#compute and print the mean cross validation score
	model = model_dispatcher.models[model_name]
	cvscore = cross_val_score(model, x_train, y_train, cv=cv)
	print("Model: {}, mean_crossvalscore: {}".format(model_name, cvscore.mean()))
	
	#fit the model on training data and compute accuracy score
	model.fit(x_train, y_train)
	preds = model.predict(x_test)
	accscore = accuracy_score(y_test, preds)
	print("Model: {}, accuracy_score: {}".format(model_name, accscore))
	
	#save the model to the folder specified by OUTPUT_PATH
	joblib.dump(model, os.path.join(config.OUTPUT_PATH, "model_{}.bin".format(model_name)))
	
	return None
	
	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str)
	parser.add_argument('--cv', type=int)
	args = parser.parse_args()
	train_test_model(model_name=args.model_name, cv=args.cv)

