import tensorflow as tf
import pandas as pd
import numpy as np


def preprocess_data(csv_path):
    df_raw_csv = pd.read_csv(csv_path, header=None)
    missing_value_indices = []
    for idx, row in df_raw_csv.iterrows():
        if row.isin(['?']).any():
            missing_value_indices.append(
                idx)  # only six instances has missing values, so getting rid of them to reduce noise
    df_raw_csv = df_raw_csv.drop(df_raw_csv.index[missing_value_indices])
    data_mat = df_raw_csv.values.astype(np.float)
    x = data_mat[:, 0:13]
    x = tf.math.l2_normalize(
        x, axis=0, epsilon=1e-12, name=None
    ).numpy()
    y = data_mat[:, 13]
    y = tf.keras.utils.to_categorical(y)
    return x, y
