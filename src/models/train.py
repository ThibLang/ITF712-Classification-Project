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
    print("Start:", datetime.now().strftime("%H:%M:%S"))

    s_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    clf_list = [
        LogisticRegressionClassifier(s_k_fold),
        SVMClassifier(s_k_fold),
        RFClassifier(s_k_fold),
        MLP(s_k_fold),
        KN(s_k_fold),
        DT(s_k_fold)]

    clf_optimized_list = [
        LogisticRegressionClassifier(s_k_fold),
        SVMClassifier(s_k_fold),
        RFClassifier(s_k_fold),
        MLP(s_k_fold),
        KN(s_k_fold),
        DT(s_k_fold)]

    data = pd.read_csv(os.path.join(data_path, 'training_data.csv'))
    labels = pd.read_csv(os.path.join(data_path, 'training_labels.csv'))

    for clf, clf_o in zip(clf_list, clf_optimized_list):
        clf.cross_validate(data, labels, optimized=False)
        clf_o.cross_validate(data, labels, optimized=True)

    for clf, clf_o in zip(clf_list, clf_optimized_list):
        display_score(clf.name, clf.get_score())
        display_score(clf_o.name + '_o', clf.get_score())

    print("End:", datetime.now().strftime("%H:%M:%S"))


if __name__ == '__main__':
    project_root_dir = Path(os.path.abspath('')).resolve().parents[1]

    train(os.path.join(project_root_dir, 'data', 'processed'))
