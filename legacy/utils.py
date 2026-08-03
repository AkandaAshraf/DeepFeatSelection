import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import json


def load_model(model_path):
    file_json = os.path.join(model_path, 'model.json')
    file_h5 = os.path.join(model_path, 'model.h5')

    json_file = open(file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'tf': tf})
    model.load_weights(file_h5)
    return model


def serialise_model(model, folder, exp, logs=None, history=None, save_structure_only=False):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    model_json = model.to_json()
    with open(folder + '/model.json', "w") as json_file:
        json_file.write(model_json)

    if not save_structure_only:
        model.save_weights(folder + '/model.h5')

        if history is not None:
            json.dump(history.params, open(folder + '/model_params.json', 'w'))
            json.dump(str(history.history), open(folder + '/model_history.json', 'w'))
    if logs is not None:
        # if 'val_f1_score' in logs:
        #     val_f1_score = list(logs['val_f1_score'].astype(float)) # f1 score is a numpy array which is not json serialisable
        #     del logs['val_f1_score']
        #     logs['val_f1_score'] = val_f1_score
        # if 'f1_score' in logs:
        #     f1_score = list(logs['f1_score'].astype(float))  # f1 score is a numpy array which is not json serialisable
        #     del logs['f1_score']
        #     logs['f1_score'] = f1_score

        if os.path.isfile(folder + '/logs.json'):
            with  open(folder + '/logs.json', 'r') as f:
                existing_json = json.load(f)
                if exp in existing_json:
                    del existing_json[exp]
                existing_json[exp] = logs
                logs = existing_json
                f.close()
        else:
            exp_logs = {}
            exp_logs[exp] = logs
            logs = exp_logs

        json.dump(logs, open(folder + '/logs.json', 'w'))

    print("saved model to disk")

    return folder


class ModelSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_output_dir, exp, save_only_best=True, monitor='val_loss', if_max=False):
        super().__init__()
        self.model_output_dir = model_output_dir
        self.prior_monitor_val = None
        self.save_only_best = save_only_best
        self.monitor = monitor
        self.if_max = if_max
        self.exp = exp
        if not os.path.isdir(self.model_output_dir):
            os.makedirs(self.model_output_dir)

    def on_epoch_end(self, epoch, logs=None):
        if self.save_only_best:
            if self.prior_monitor_val is None:
                self.prior_monitor_val = logs.get(self.monitor)

            else:
                if not self.if_max:
                    if logs.get(self.monitor) <= self.prior_monitor_val:
                        print('saving model, improved ' + self.monitor + ': ' + str(
                            self.prior_monitor_val - logs.get(self.monitor)))
                        logs['epoch'] = epoch
                        serialise_model(self.model, folder=self.model_output_dir, save_structure_only=False, logs=logs,
                                        exp=self.exp)

                    else:
                        if logs.get(self.monitor) >= self.prior_monitor_val:
                            print('saving model, improved ' + self.monitor + ': ' + str(
                                logs.get(self.monitor) - self.prior_monitor_val))
                            logs['epoch'] = epoch
                            serialise_model(self.model, folder=self.model_output_dir, save_structure_only=False,
                                            logs=logs, exp=self.exp)

        else:
            serialise_model(self.model, folder=self.model_output_dir,
                            save_structure_only=False, logs=logs, exp=self.exp)
