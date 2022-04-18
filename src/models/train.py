import os
import pandas as pd
import numpy as np

from pathlib import Path

from src.models.LogisticRegression import LogisticRegressionClassifier
from src.models.SVM import SVMClassifier
from src.models.RandomForest import RFClassifier
from src.models.MLP import MLP
from src.models.KN import KN
from src.models.DecisionTree import DT
from sklearn.model_selection import StratifiedKFold

from datetime import datetime


def display_score(name, score):
    score_string = name + ': '
    for key in score:
        score_string += key + '={:.4f}'.format(score[key]) + '\t'

    print(score_string)


def train(data_path):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    s_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logistic_regression = LogisticRegressionClassifier(s_k_fold)
    svm = SVMClassifier(s_k_fold)
    rf = RFClassifier(s_k_fold)
    mlp = MLP(s_k_fold)
    kn = KN(s_k_fold)
    dt = DT(s_k_fold)

    data = pd.read_csv(os.path.join(data_path, 'training_data.csv'))
    labels = pd.read_csv(os.path.join(data_path, 'training_labels.csv'))

    logistic_regression.cross_validate(data, labels, pre_trained=False)
    display_score(logistic_regression.get_score())
    logistic_regression.optimize(data, labels)
    logistic_regression.cross_validate(data, labels, pre_trained=True)
    display_score(logistic_regression.get_score())

    svm.cross_validate(data, labels, pre_trained=False)
    display_score(svm.get_score())
    svm.optimize(data, labels)
    svm.cross_validate(data, labels, pre_trained=True)
    display_score(svm.get_score())

    dt.cross_validate(data, labels, pre_trained=False)
    display_score(dt.get_score())
    dt.optimize(data, labels)
    dt.cross_validate(data, labels, pre_trained=True)
    display_score(dt.get_score())

    mlp.cross_validate(data, labels, pre_trained=False)
    display_score(mlp.get_score())
    mlp.optimize(data, labels)
    mlp.cross_validate(data, labels, pre_trained=True)
    display_score(mlp.get_score())

    kn.cross_validate(data, labels, pre_trained=False)
    display_score(kn.get_score())
    kn.optimize(data, labels)
    kn.cross_validate(data, labels, pre_trained=True)
    display_score(kn.get_score())
    now = datetime.now()

    rf.cross_validate(data, labels, pre_trained=False)
    display_score(rf.get_score())
    rf.optimize(data, labels)
    rf.cross_validate(data, labels, pre_trained=True)
    display_score(rf.get_score())

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("End =", current_time)


if __name__ == '__main__':
    project_root_dir = Path(os.path.abspath('')).resolve().parents[1]

    train(os.path.join(project_root_dir, 'data', 'processed'))
