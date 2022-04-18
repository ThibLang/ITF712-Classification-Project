import numpy as np
import json
from sklearn.neural_network import MLPClassifier
from src.models.Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV


class MLP(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.mlp = None
        self.name = "MLP"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.mlp = MLPClassifier()
        else:
            with open(self.name + '_hyp', 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            self.mlp = MLPClassifier(hidden_layer_sizes=hyp['hidden_layer_sizes'],
                                     solver=hyp['solver'],
                                     alpha=hyp['alpha'],
                                     learning_rate=hyp['learning_rate'],
                                     early_stopping=hyp['early_stopping'])

    def cross_validate(self, data, labels, pre_trained=False):
        self.initialize_classifier(pre_trained)

        self.start_training()
        for training_index, test_index in self.fold.split(data, labels):
            training_data, test_data = data.values[training_index], data.values[test_index]
            training_label, test_label = labels.values[training_index], labels.values[test_index]

            self.mlp.fit(training_data, np.ravel(training_label))

            y_pred = self.mlp.predict(test_data)
            y_proba = self.mlp.predict_proba(test_data)
            self.compute_test_results(test_label, y_pred, y_proba)

        self.end_training()

    def optimize(self, data, labels):
        self.initialize_classifier()
        self.print('Start optimization')

        hyp_grid = [
            {'hidden_layer_sizes': [100, (100, 100), 500, (500, 500), 1000, 2000],
             'solver': ['lbfgs', 'sgd', 'adam'],
             'alpha':np.logspace(-3, 10, 8),
             'learning_rate': ['adaptive'],
             'early_stopping': [True]}
        ]

        search = RandomizedSearchCV(self.mlp,
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
            json.dump(result.best_params_, fp)

        self.print('end optimization')

