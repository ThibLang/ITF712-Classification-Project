import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import logging

RANDOM_STATE = 42


class LeafClassificationPipeline:
    def __init__(self, config_file):
        self.logger = logging.getLogger(__name__)

        # Parse the configuration file
        if isinstance(config_file, str):
            with open(config_file, errors='ignore') as f:
                self.config = yaml.safe_load(f)  # load hyps dict

        self.logger.info('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in self.config.items()))

        if self.config['scaler'] == 'Standard':
            scaler = ('scaler', StandardScaler())
        elif self.config['scaler'] == 'MinMax':
            scaler = ('scaler', MinMaxScaler())
        elif self.config['scaler'] == 'Normalisation':
            scaler = ('scaler', Normalizer())
        else:
            # Use the standard scaler by default
            scaler = ('scaler', StandardScaler)

        if self.config['encoder'] == 'Label':
            self.encoder = LabelEncoder()
        else:
            self.encoder = OneHotEncoder()

        self.split_ratio = self.config['ratio']
        self.splitter = StratifiedShuffleSplit(n_splits=1,
                                               test_size= 1.0 - self.split_ratio/100.0,
                                               random_state=RANDOM_STATE)

        self.transformation_pipeline = Pipeline([scaler])

    def fit_transform(self, input_training_data, input_test_data):
        self.logger.info('Transform the dataset')
        train_set = self.transformation_pipeline.fit_transform(input_training_data)
        test_set = self.transformation_pipeline.transform(input_test_data)

        return train_set, test_set

    def encode_label(self, labels):
        return self.encoder.fit_transform(labels)

    def split_training_test_data(self, input_data, input_labels):
        self.logger.info('Split the dataset')
        for train_index, test_index in self.splitter.split(input_data, input_labels):
            train_set, test_set = input_data.loc[train_index], input_data.loc[test_index]
            train_labels, test_labels = input_labels[train_index], input_labels[test_index]

        return train_set, train_labels, test_set, test_labels
