from utils import load_model
import numpy as np


def get_feat_weights(model_path, attribute_names):
    model = load_model(model_path)
    weights = []
    for attribute in attribute_names:
        weights.append(model.get_layer(attribute).weights[0].numpy()[0])
    weights = np.stack(weights)  # create 1d np array
    weights = weights / np.sum(weights)  # normalise weights
    return weights.flatten()
