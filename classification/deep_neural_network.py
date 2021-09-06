import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from plotting import *


def deep_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Deep Neural Network created for classification input data.

    References
    ----------
    [1] https://keras.io/

    Parameters
    ----------
    X_train : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y_train : ndarray
        Array of labels belongs to input X_train data.

    X_test : ndarray {default: None}
        Array of data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ]. Could be None.

    y_test : ndarray {default: None}
        Array of labels belongs to X_test data. Could be None.

    Returns
    -------
    dnn
        Trained classifier ready to predict.
    """
    # _check_deep_network_params(X_train, y_train)

    tf.random.set_seed(1)
    num_epochs = 19
    batch_size = 1
    steps_per_epoch = int(np.ceil(len(y_train) / batch_size))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=10,
                                    activation='softplus'))
    model.add(tf.keras.layers.Dense(units=15,
                                    activation='tanh'))
    model.add(tf.keras.layers.Dense(units=50,
                                    activation='softsign'))
    model.add(tf.keras.layers.Dense(units=6,
                                    activation='softmax'))
    model.build(input_shape=(None, 2))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              steps_per_epoch=steps_per_epoch)

    if X_test is not None and y_test is not None and len(X_test) == len(y_test):
        y_pred = model.predict(X_test)
        y_pred = list(map(lambda item: np.argmax(item), y_pred))
        print(f"Deep Neural Network test accuracy: {accuracy_score(y_test, y_pred)}")

    return model


def _check_deep_network_params(X: np.ndarray, y: np.ndarray):
    """
    Check the number of layers, types of layers, activation functions etc.
    needed in a deep neural network. Show results on plots.

    Parameters
    ----------
    X : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y : ndarray
        Array of labels belongs to input X data.
    """
    tf.random.set_seed(1)
    num_epochs = 19
    batch_size = 1
    steps_per_epoch = int(np.ceil(len(y) / batch_size))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units=10,
                                    activation='softplus'))
    model.add(tf.keras.layers.Dense(units=15,
                                    activation='tanh'))
    model.add(tf.keras.layers.Dense(units=50,
                                    activation='softsign'))
    model.add(tf.keras.layers.Dense(units=6,
                                    activation='softmax'))
    model.build(input_shape=(None, 2))

    print(model.summary())

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        steps_per_epoch=steps_per_epoch)

    plot_decision_regions(model, X, y)
    plot_learning_history(history.history)
