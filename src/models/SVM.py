import numpy as np
import json
from sklearn.svm import SVC
from src.models.Classifier import Classifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform


class SVMClassifier(Classifier):
    def __init__(self, fold):
        super().__init__()
        self.fold = fold
        self.svm = None
        self.name = "SVM"

        self.print('Creating')

    def initialize_classifier(self, pre_trained=False):
        self.print('Initialization')
        if not pre_trained:
            self.svm = SVC(probability=True)
        else:
            with open(self.name + '_hyp', 'r') as fp:
                hyp = json.load(fp)

                hyp_string = ''
                for key in hyp:
                    hyp_string += key + ':' + str(hyp[key]) + ' '
                self.print(hyp_string)

            if hyp['kernel'] == 'linear' or hyp['kernel'] == 'sigmoid':
                self.svm = SVC(kernel=hyp['kernel'], C=hyp['C'], probability=True)

            elif hyp['kernel'] == 'poly':
                self.svm = SVC(kernel=hyp['kernel'], C=hyp['C'], degree=hyp['degree'], probability=True)

            elif hyp['kernel'] == 'rbf':
                self.svm = SVC(kernel=hyp['kernel'], C=hyp['C'], gamma=hyp['gamma'], probability=True)

    def cross_validate(self, data, labels, pre_trained=False):
        self.initialize_classifier(pre_trained)

        self.start_training()
        for training_index, test_index in self.fold.split(data, labels):
            training_data, test_data = data.values[training_index], data.values[test_index]
            training_label, test_label = labels.values[training_index], labels.values[test_index]

            self.svm.fit(training_data, np.ravel(training_label))

            y_pred = self.svm.predict(test_data)
            y_proba = self.svm.predict_proba(test_data)
            self.compute_test_results(test_label, y_pred, y_proba)

        self.end_training()

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

        search = RandomizedSearchCV(self.svm,
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

