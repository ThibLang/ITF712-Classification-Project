import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV


class DT(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.DT = None
        self.name = "Decision Tree"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.DT = DecisionTreeClassifier()
        else:
            with open(self.name + '_hyp', 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            self.DT = DecisionTreeClassifier(splitter=hyp['splitter'],
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

            self.DT.fit(training_data, np.ravel(training_label))

            y_pred = self.DT.predict(test_data)
            y_proba = self.DT.predict_proba(test_data)
            self.compute_test_results(test_label, y_pred, y_proba)

        self.end_training()

    def optimize(self, data, labels):
        self.initialize_classifier()
        self.print('Start optimization')

        hyp_grid = [
            {'splitter': ['best', 'random'],
             'criterion': ['gini', 'entropy'],
             'max_depth':np.linspace(1, 20, num=10, dtype=int),
             'min_samples_split': np.linspace(1, 500, num=20, dtype=int),
             'min_samples_leaf': np.linspace(1, 500, num=20, dtype=int),
             'max_features': ['sqrt', 'log2', None],
             'random_state': [42]}
        ]

        search = RandomizedSearchCV(self.DT,
                                    hyp_grid,
                                    n_iter=200,
                                    scoring='neg_log_loss',
                                    cv=self.fold,
                                    random_state=42,
                                    verbose=True,
                                    n_jobs=-1)

        result = search.fit(data.values, np.ravel(labels.values))

        # Saving the results in a jon file
        with open(self.name + '_hyp', 'w') as fp:
            result.best_params_['max_depth'] = int(result.best_params_['max_depth'])
            result.best_params_['min_samples_split'] = int(result.best_params_['min_samples_split'])
            result.best_params_['min_samples_leaf'] = int(result.best_params_['min_samples_leaf'])
            json.dump(result.best_params_, fp)

        self.print('end optimization')

