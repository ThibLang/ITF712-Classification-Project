import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from src.models.Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

# This code section avoid to be flooded with ConvergenceWarning from the randomizeSearch
import sys
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
###


class LogisticRegressionClassifier(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.clf = None
        self.name = "Logistic Regression"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.clf = LogisticRegression()
        else:
            with open(self.name + '_hyp', 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            self.clf = LogisticRegression(C=hyp['C'], penalty=hyp['penalty'], solver=hyp['solver'])

    def optimize(self, data, labels):
        self.initialize_classifier()
        self.print('Start optimization')

        hyp_grid = [
            {'solver': ['newton-cg'], 'penalty': ['l2'], 'C': loguniform(1e-5, 1000)},
            {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': loguniform(1e-5, 1000)},
            {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': loguniform(1e-5, 1000)},
            {'solver': ['sag'], 'penalty': ['l2'], 'C': loguniform(1e-5, 1000)},
            {'solver': ['saga'], 'penalty': ['elasticnet', 'l1', 'l2'], 'C': loguniform(1e-5, 1000)}
        ]

        search = RandomizedSearchCV(self.clf,
                                    hyp_grid,
                                    n_iter=100,
                                    scoring='neg_log_loss',
                                    cv=self.fold,
                                    random_state=42,
                                    n_jobs=-1)

        result = search.fit(data.values, np.ravel(labels.values))

        # Saving the results in a jon file
        with open(self.name + '_hyp', 'w') as fp:
            json.dump(result.best_params_, fp)

        self.print('end optimization')

