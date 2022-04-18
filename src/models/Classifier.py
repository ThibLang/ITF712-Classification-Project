from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss
import numpy as np


class Classifier:
    def __init__(self):
        """
        Initialization
        : return: nothing
        """
        self.f1_score = []
        self.precision_score = []
        self.recall_score = []
        self.accuracy_score = []
        self.log_loss = []

        self.results = None

    def start_training(self):
        """
        Call at the start of the training to initialize the score
        : return: nothing
        """
        self.f1_score = []
        self.precision_score = []
        self.recall_score = []
        self.accuracy_score = []
        self.log_loss = []
        self.results = None

    def compute_test_results(self, y_true, y_pred, y_proba):
        """
        Compute metrics score at each step of a cross validation
        :param y_true: true label of each element of the test set
        :param y_pred: predicted label for each element of the test set
        :param y_proba: predicted probality for each class for each element of the test set
        :return: nothing
        """

        unique_class = np.unique(y_true)

        self.f1_score.append(f1_score(y_true=y_true, y_pred=y_pred, average='weighted', labels=unique_class))
        self.precision_score.append(precision_score(y_true=y_true, y_pred=y_pred, average='weighted', labels=unique_class))
        self.recall_score.append(recall_score(y_true=y_true, y_pred=y_pred, average='weighted', labels=unique_class))
        self.accuracy_score.append(accuracy_score(y_true=y_true, y_pred=y_pred))
        self.log_loss.append(log_loss(y_true=y_true, y_pred=y_proba, labels=unique_class))

    def end_training(self):
        """
        Compute the mean for each metrics and return them
        :return: nothing
        """
        mean_f1 = np.mean(self.f1_score)
        mean_precision = np.mean(self.precision_score)
        mean_recall = np.mean(self.recall_score)
        mean_accuracy = np.mean(self.accuracy_score)
        mean_log_loss = np.mean(self.log_loss)
        self.results = {'f1': mean_f1,
                        'precision': mean_precision,
                        'recall': mean_recall,
                        'accuracy': mean_accuracy,
                        'log_loss': mean_log_loss}

    def get_score(self):
        """
        Get the results
        :return: dict: {f1, precision, recall, accuracy, log_loss}
        """
        return self.results

    def print(self, string):
        """
        Print with the name of the classifier
        :return: nothing
        """
        print(self.name + ':' + string)



