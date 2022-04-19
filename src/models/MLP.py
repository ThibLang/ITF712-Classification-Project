import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from src.models.Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV


class MLP(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.clf = None
        self.name = "MLP"

        self.print('Creating')

    def initialize_classifier(self, optimized=False):
        self.print('Initialization')
        if not optimized:
            self.clf = MLPClassifier()
        else:
            with open(self.get_config_file_path(), 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            self.clf = MLPClassifier(hidden_layer_sizes=hyp['hidden_layer_sizes'],
                                     solver=hyp['solver'],
                                     alpha=hyp['alpha'],
                                     learning_rate=hyp['learning_rate'],
                                     early_stopping=hyp['early_stopping'])

    def optimize(self, data, labels):
        self.initialize_classifier()
        self.print('Start optimization')

        hyp_grid = [
            {'hidden_layer_sizes': [100, (100, 100), 500, (500, 500)],
             'solver': ['lbfgs', 'adam'],
             'alpha':np.logspace(-3, 10, 8),
             'learning_rate': ['adaptive'],
             'early_stopping': [True]}
        ]

        search = RandomizedSearchCV(self.clf,
                                    hyp_grid,
                                    n_iter=50,
                                    scoring='neg_log_loss',
                                    cv=self.fold,
                                    random_state=42,
                                    verbose=True,
                                    n_jobs=-1)

        result = search.fit(data.values, np.ravel(labels.values))

        # Saving the results in a jon file
        with open(self.get_config_file_path(), 'w') as fp:
            json.dump(result.best_params_, fp)

        self.print('end optimization')

