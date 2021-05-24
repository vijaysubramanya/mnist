'''
The dictionary ``models`` contains the classifier object
for decision tree, random forest, and logistic regression models.
'''

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
	'decision_tree': DecisionTreeClassifier(),
	'rf': RandomForestClassifier(),
	'log_reg': LogisticRegression()
}
