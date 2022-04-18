import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from src.models.Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV


class KN(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.clf = None
        self.name = "KN"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.clf = KNeighborsClassifier()
        else:
            with open(self.get_config_file_path(), 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            self.clf = KNeighborsClassifier(n_neighbors=hyp['n_neighbors'],
                                            weights=hyp['weights'],
                                            algorithm=hyp['algorithm'],
                                            leaf_size=hyp['leaf_size'],
                                            p=hyp['p'], n_jobs=-1)

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

        search = RandomizedSearchCV(self.clf,
                                    hyp_grid,
                                    n_iter=500,
                                    scoring='neg_log_loss',
                                    cv=self.fold,
                                    random_state=42,
                                    verbose=True,
                                    n_jobs=-1)

        result = search.fit(data.values, np.ravel(labels.values))

        # Saving the results in a jon file
        with open(self.get_config_file_path(), 'w') as fp:
            result.best_params_['leaf_size'] = int(result.best_params_['leaf_size'])
            json.dump(result.best_params_, fp)

        self.print('end optimization')

