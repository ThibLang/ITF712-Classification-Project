import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from Classifier import Classifier
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
        self.logistic_regression = None
        self.name = "Logistic Regression"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.logistic_regression = LogisticRegression()
        else:
            with open(self.name + '_hyp', 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            self.logistic_regression = LogisticRegression(C=hyp['C'], penalty=hyp['penalty'], solver=hyp['solver'])

    def cross_validate(self, data, labels, pre_trained=False):
        self.initialize_classifier(pre_trained)

        self.start_training()
        for training_index, test_index in self.fold.split(data, labels):
            training_data, test_data = data.values[training_index], data.values[test_index]
            training_label, test_label = labels.values[training_index], labels.values[test_index]

            self.logistic_regression.fit(training_data, np.ravel(training_label))

            y_pred = self.logistic_regression.predict(test_data)
            y_proba = self.logistic_regression.predict_proba(test_data)
            self.compute_test_results(test_label, y_pred, y_proba)

        self.end_training()

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

        search = RandomizedSearchCV(self.logistic_regression,
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

