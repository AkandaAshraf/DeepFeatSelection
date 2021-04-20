from env_loader import GetEnvs
from get_feat_weights import get_feat_weights
from train import train
import pandas as pd
import numpy as np
import tensorflow as tf


class Experiment:
    def __init__(self, input_csv_data, output_csv_path, attribute_names, num_models, model_output_path='Models'):
        self.input_csv_data = input_csv_data
        self.output_csv_path = output_csv_path
        self.attribute_names = attribute_names
        if type(self.attribute_names) is not list and type(self.attribute_names) is not tuple:
            self.attribute_names = self._load_attributes_from_file()

        self.num_models = num_models
        self.model_output_path = model_output_path

    def _load_attributes_from_file(self):
        return list(pd.read_csv(self.attribute_names, header=None).values[0])

    def run_experiment(self):
        for i in range(self.num_models):
            tf.compat.v1.reset_default_graph()

            print('exp number: ' + str(i))

            train(attribute_names=self.attribute_names, input_csv=self.input_csv_data,
                  model_output_path=self.model_output_path, exp=i)
            output = get_feat_weights(self.model_output_path, self.attribute_names)

            with open(self.output_csv_path, 'a') as f:
                is_start = True
                for c in list(output):
                    if is_start:
                        f.write(str(c))
                        is_start = False
                    else:
                        f.write(',' + str(c))
                f.write('\n')

    def get_features_importance(self):
        mat = pd.read_csv(self.output_csv_path, header=None).values
        avg_feat_importance = np.average(mat, axis=0)
        avg_feat_importance = avg_feat_importance / avg_feat_importance.sum()
        avg_feat_importance_dict = {}
        for i in range(len(self.attribute_names)):
            avg_feat_importance_dict[self.attribute_names[i]] = avg_feat_importance[i]

        avg_feat_importance_dict = dict(
            sorted(avg_feat_importance_dict.items(), reverse=True, key=lambda item: item[1]))
        print('the output feature weights for all the models can be found in: ' + self.output_csv_path)
        print('calculated feature weights (sorted), higher is better: ')
        print(avg_feat_importance_dict)
        return avg_feat_importance_dict


if __name__ == '__main__':
    num_models = int(input(
        'Please enter an integer value specifying the number of models (a large number will take longer but will produce more accurate result) to train: '))

    attribute_names = ['age',
                       'sex',
                       'cp',
                       'trestbps',
                       'chol',
                       'fbs',
                       'restecg',
                       'thalach',
                       'exang',
                       'oldpeak',
                       'slope',
                       'ca',
                       'thal']
    output_weights_csv = r'ExpOutput\output_weights.csv'
    input_csv_data = r'Data/processed.cleveland.data'
    model_output_path = 'Models'
    env_obj = GetEnvs(['DATA_FILE', 'EXP_OUTPUT_CSV_FILE', 'OUTPUT_MODEL_DIR', 'ATTRIBUTE_FILE'])
    if env_obj.load_envs():
        exp_object = Experiment(input_csv_data=env_obj.get_env_vars('DATA_FILE'),
                                output_csv_path=env_obj.get_env_vars('EXP_OUTPUT_CSV_FILE'),
                                attribute_names=env_obj.get_env_vars('ATTRIBUTE_FILE'),
                                num_models=num_models,
                                model_output_path=env_obj.get_env_vars('OUTPUT_MODEL_DIR'))
        print('using default the following env vars')
        print(env_obj.env_dict)
    else:
        print('one more env var(s) provided, using default data and attributes..')
        exp_object = Experiment(input_csv_data=input_csv_data,
                                output_csv_path=output_weights_csv,
                                attribute_names=attribute_names,
                                num_models=num_models, model_output_path=model_output_path)

    exp_object.run_experiment()
    exp_object.get_features_importance()
