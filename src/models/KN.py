import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from src.models.Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV


class KN(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.kn = None
        self.name = "KN"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.kn = KNeighborsClassifier()
        else:
            with open(self.name + '_hyp', 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            self.kn = KNeighborsClassifier(n_neighbors=hyp['n_neighbors'],
                                           weights=hyp['weights'],
                                           algorithm=hyp['algorithm'],
                                           leaf_size=hyp['leaf_size'],
                                           p=hyp['p'], n_jobs=-1)

    def cross_validate(self, data, labels, pre_trained=False):
        self.initialize_classifier(pre_trained)

        self.start_training()
        for training_index, test_index in self.fold.split(data, labels):
            training_data, test_data = data.values[training_index], data.values[test_index]
            training_label, test_label = labels.values[training_index], labels.values[test_index]

            self.kn.fit(training_data, np.ravel(training_label))

            y_pred = self.kn.predict(test_data)
            y_proba = self.kn.predict_proba(test_data)
            self.compute_test_results(test_label, y_pred, y_proba)

        self.end_training()

    def optimize(self, data, labels):
        self.initialize_classifier()
        self.print('Start optimization')

        hyp_grid = [
            {'n_neighbors': [1, 2, 3, 4, 5, 6],
             'weights': ['uniform', 'distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree'],
             'leaf_size': np.linspace(200, 600, num=50, dtype=int),
             'p': [1, 2, 3]}
        ]

        search = RandomizedSearchCV(self.kn,
                                    hyp_grid,
                                    n_iter=500,
                                    scoring='neg_log_loss',
                                    cv=self.fold,
                                    random_state=42,
                                    verbose=True,
                                    n_jobs=-1)

        result = search.fit(data.values, np.ravel(labels.values))

        # Saving the results in a jon file
        with open(self.name + '_hyp', 'w') as fp:
            result.best_params_['leaf_size'] = int(result.best_params_['leaf_size'])
            json.dump(result.best_params_, fp)

        self.print('end optimization')

