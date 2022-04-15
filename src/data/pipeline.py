import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, LabelEncoder, OneHotEncoder
import logging

class TransformationPipeline:
    def __init__(self, config_file):
        logger = logging.getLogger(__name__)

        # Parse the configuration file
        if isinstance(config_file, str):
            with open(config_file, errors='ignore') as f:
                self.config = yaml.safe_load(f)  # load hyps dict

        logger.info('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in self.config.items()))

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

        self.transformation_pipeline = Pipeline([scaler])

    def fit_transform(self, input_data):
        return self.transformation_pipeline.fit_transform(input_data)

    def encode_label(self, labels):
        return self.encoder.fit_transform(labels)
