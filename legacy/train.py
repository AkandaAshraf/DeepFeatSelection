from early_stop_callback import EarlyStoppingModified
from model import get_model
from preprocess_data import preprocess_data
import numpy as np


def split_array(arr, num_splits):
    arr_list = []
    for i in range(num_splits):
        arr_list.append(np.expand_dims(arr[:, i], -1))
    return arr_list


def train(attribute_names, input_csv, model_output_path, exp):
    x, y = preprocess_data(input_csv)
    num_classes = y.shape[-1]
    y = split_array(y, num_classes)
    x = split_array(x, x.shape[-1])

    monitor_values = ["val_output_class_0_precision",
                      "val_output_class_0_recall",
                      "val_output_class_1_precision",
                      "val_output_class_1_recall",
                      "val_output_class_2_precision",
                      "val_output_class_2_recall",
                      "val_output_class_3_precision",
                      "val_output_class_3_recall",
                      "val_output_class_4_precision",
                      "val_output_class_4_recall"]

    model = get_model(attribute_names=attribute_names, lr=0.0001, num_output_classes=num_classes)
    # early_stopper_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', min_delta=0, patience=10, verbose=0,
    #     mode='min', baseline=None, restore_best_weights=True
    # )
    early_stopper_callback = EarlyStoppingModified(model_output_dir=model_output_path, exp=exp,
                                                   monitor='val_loss', min_delta=0, patience=15, verbose=0,
                                                   mode='min', baseline=None, restore_best_weights=True
                                                   )  # this callback has been modified, taken from tensorflow. It has been modified to use a list of monitor value and also implements model saver callback

    model.fit(x, y, epochs=2000, shuffle=True, validation_split=0.5, batch_size=64,
              callbacks=[early_stopper_callback])
