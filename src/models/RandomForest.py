import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform


class RFClassifier(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.rf = None
        self.name = "Random Forest"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.rf = RandomForestClassifier()
        else:
            with open(self.name + '_hyp', 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            self.rf = RandomForestClassifier(n_estimators=hyp['n_estimators'],
                                             criterion=hyp['criterion'],
                                             max_depth=hyp['max_depth'],
                                             min_samples_split=hyp['min_samples_split'],
                                             min_samples_leaf=hyp['min_samples_leaf'],
                                             max_features=hyp['max_features'],
                                             random_state=hyp['random_state'])

    def cross_validate(self, data, labels, pre_trained=False):
        self.initialize_classifier(pre_trained)

        self.start_training()
        for training_index, test_index in self.fold.split(data, labels):
            training_data, test_data = data.values[training_index], data.values[test_index]
            training_label, test_label = labels.values[training_index], labels.values[test_index]

            self.rf.fit(training_data, np.ravel(training_label))

            y_pred = self.rf.predict(test_data)
            y_proba = self.rf.predict_proba(test_data)
            self.compute_test_results(test_label, y_pred, y_proba)

        self.end_training()

    def optimize(self, data, labels):
        self.initialize_classifier()
        self.print('Start optimization')

        hyp_grid = [
            {'n_estimators': np.linspace(1000, 10000, num=10, dtype=int),
             'criterion': ['gini', 'entropy'],
             'max_depth':np.linspace(1, 100, num=10, dtype=int),
             'min_samples_split': [2, 4, 6, 8],
             'min_samples_leaf': [1, 2, 4],
             'max_features': ['sqrt', 'log2', None],
             'random_state': [42]}
        ]

        search = RandomizedSearchCV(self.rf,
                                    hyp_grid,
                                    n_iter=50,
                                    scoring='neg_log_loss',
                                    cv=self.fold,
                                    random_state=42,
                                    verbose=True,
                                    n_jobs=-1)

        result = search.fit(data.values, np.ravel(labels.values))

        # Saving the results in a jon file
        with open(self.name + '_hyp', 'w') as fp:
            result.best_params_['n_estimators'] = int(result.best_params_['n_estimators'])
            result.best_params_['max_depth'] = int(result.best_params_['max_depth'])
            json.dump(result.best_params_, fp)

        self.print('end optimization')

