import numpy as np
import json
from sklearn.svm import SVC
from src.models.Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV


class SVMClassifier(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.clf = None
        self.name = "SVM"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.clf = SVC(probability=True)
        else:
            with open(self.name + '_hyp', 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            if hyp['kernel'] == 'linear' or hyp['kernel'] == 'sigmoid':
                self.clf = SVC(kernel=hyp['kernel'], C=hyp['C'], probability=True)

            elif hyp['kernel'] == 'poly':
                self.clf = SVC(kernel=hyp['kernel'], C=hyp['C'], degree=hyp['degree'], probability=True)

            elif hyp['kernel'] == 'rbf':
                self.clf = SVC(kernel=hyp['kernel'], C=hyp['C'], gamma=hyp['gamma'], probability=True)

    def optimize(self, data, labels):
        self.initialize_classifier()
        self.print('Start optimization')

        hyp_grid = [
            {'kernel': ['linear', 'sigmoid'],
             'C': np.logspace(-3, 100, 5),
             'probability':[True]},
            {'kernel': ['poly'],
             'C': np.logspace(-3, 100, 5),
             'probability': [True],
             'degree': [1, 3, 5, 10],
             'gamma': ['scale', 'auto']},
            {'kernel': ['rbf', ],
             'C': np.logspace(-3, 100, 5),
             'probability': [True],
             'gamma': np.logspace(-3, 100, 5)}
        ]

        search = RandomizedSearchCV(self.clf,
                                    hyp_grid,
                                    n_iter=5,
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

