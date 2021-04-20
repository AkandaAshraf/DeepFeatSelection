import tensorflow as tf
from tensorflow.keras.regularizers import l2


def get_f1_loss(y, y_hat):
    ## this loss function is taken from https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d

    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0)  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels

    return macro_cost


def get_model(attribute_names, num_output_classes, lr=0.001):
    inputs = []
    for i in range(len(attribute_names)):
        inputs.append(tf.keras.layers.Input(1))

    feature_layer_outputs = []

    for i in range(len(attribute_names)):
        feature_layer_outputs.append(tf.keras.layers.Dense(1,
                                                           name=attribute_names[i],
                                                           activation='relu',
                                                           kernel_initializer=tf.keras.initializers.GlorotUniform(),

                                                           kernel_constraint=tf.keras.constraints.NonNeg(),
                                                           use_bias=False)(
            inputs[
                i]))  # constraining the weight to be non-neg using tf.keras.constraints.NonNeg() and intialise weights as 1, which will allow the data to be passed directly

    conc_output = tf.keras.layers.Concatenate(axis=-1)(feature_layer_outputs)

    H = tf.keras.layers.Dense(128, name='fc_1', activation='relu', kernel_regularizer=l2(0.001),
                              bias_regularizer=l2(0.001))(conc_output)
    H = tf.keras.layers.GaussianNoise(0.005)(H)
    H = tf.keras.layers.Dropout(0.5)(H)
    H = tf.keras.layers.Dense(128, name='fc_2', activation='relu', kernel_regularizer=l2(0.001),
                              bias_regularizer=l2(0.001))(H)
    H = tf.keras.layers.GaussianNoise(0.005)(H)
    H = tf.keras.layers.Dropout(0.5)(H)
    H = tf.keras.layers.Dense(128, name='fc_3', activation='relu', kernel_regularizer=l2(0.001),
                              bias_regularizer=l2(0.001))(H)
    H = tf.keras.layers.Dropout(0.5)(H)
    outputs = []

    for i in range(num_output_classes):
        outputs.append(tf.keras.layers.Dense(1, name='output_class_' + str(i), activation='sigmoid')(H))

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss=get_f1_loss,
                  optimizer=opt,
                  metrics=[tf.metrics.Precision(), tf.metrics.Recall()]
                  )

    model.summary()
    return model
