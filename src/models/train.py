import os
import pandas as pd
import numpy as np

from pathlib import Path

from LogisticRegression import LogisticRegressionClassifier
from sklearn.model_selection import StratifiedKFold


def display_score(score):
    for key in score:
        print(key, ':', score[key])


def train(data_path):
    s_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logistic_regression = LogisticRegressionClassifier(s_k_fold)

    data = pd.read_csv(os.path.join(data_path, 'training_data.csv'))
    labels = pd.read_csv(os.path.join(data_path, 'training_labels.csv'))

    logistic_regression.cross_validate(data, labels, pre_trained=True)
    display_score(logistic_regression.get_score())

    logistic_regression.optimize(data, labels)
    logistic_regression.cross_validate(data, labels, pre_trained=True)
    display_score(logistic_regression.get_score())


if __name__ == '__main__':
    project_root_dir = Path(os.path.abspath('')).resolve().parents[1]

    train(os.path.join(project_root_dir, 'data', 'processed'))
