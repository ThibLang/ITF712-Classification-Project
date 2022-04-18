import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from src.models.Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV


class DT(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.clf = None
        self.name = "Decision Tree"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.clf = DecisionTreeClassifier()
        else:
            with open(self.get_config_file_path(), 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            self.clf = DecisionTreeClassifier(splitter=hyp['splitter'],
                                              criterion=hyp['criterion'],
                                              max_depth=hyp['max_depth'],
                                              min_samples_split=hyp['min_samples_split'],
                                              min_samples_leaf=hyp['min_samples_leaf'],
                                              max_features=hyp['max_features'],
                                              random_state=hyp['random_state'])

    def optimize(self, data, labels):
        self.initialize_classifier()
        self.print('Start optimization')

        hyp_grid = [
            {'splitter': ['best', 'random'],
             'criterion': ['gini', 'entropy'],
             'max_depth': [None],
             'min_samples_split': np.linspace(200, 600, num=20, dtype=int),
             'min_samples_leaf': np.linspace(25, 75, num=20, dtype=int),
             'max_features': ['sqrt', 'log2', None],
             'random_state': [42]}
        ]

        search = RandomizedSearchCV(self.clf,
                                    hyp_grid,
                                    n_iter=200,
                                    scoring='neg_log_loss',
                                    cv=self.fold,
                                    random_state=42,
                                    verbose=True,
                                    n_jobs=-1)

        result = search.fit(data.values, np.ravel(labels.values))

        # Saving the results in a jon file
        with open(self.get_config_file_path(), 'w') as fp:
            result.best_params_['min_samples_split'] = int(result.best_params_['min_samples_split'])
            result.best_params_['min_samples_leaf'] = int(result.best_params_['min_samples_leaf'])
            json.dump(result.best_params_, fp)

        self.print('end optimization')

