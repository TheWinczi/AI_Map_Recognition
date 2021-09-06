import numpy as np
import tensorflow as tf
import pandas as pd
import os


def neural_network(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Deep Neural Network regressor used for regression.

    References
    ----------
    [1] https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor

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
    regressor
        Trained regressor ready to predict.
    """

    df_train = _cast_to_dataframe(X_train, y_train)
    df_test = _cast_to_dataframe(X_test, y_test)

    numeric_cols_names = ['x', 'y']
    numeric_features = []
    for col_name in numeric_cols_names:
        numeric_features.append(
            tf.feature_column.numeric_column(key=col_name))

    all_features_columns = (
        numeric_features
    )

    hidden_units = _check_neural_network_params(all_features_columns, df_train, df_test)
    epochs = 1000
    batch_size = 5
    total_steps = epochs * int(np.ceil(len([df_train]) / batch_size))

    regressor = tf.estimator.DNNRegressor(
        feature_columns=all_features_columns,
        hidden_units=hidden_units,
        model_dir=os.path.join("models", "dnn_regressors", "best"),
    )

    regressor.train(
        input_fn=lambda: train_input_fn(df_train, batch_size=batch_size),
        steps=total_steps
    )

    return regressor


def _check_neural_network_params(features_columns: list,
                                 df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Check all parameters needed in Deep Neural Network regressor.

    Parameters
    ----------
    features_columns : list
        List of input features columns as a list of tuples/lists.

    df_train : DataFrame
        DataFrame stores data needed to train.

    df_test : DataFrame
        DataFrame stores data needed to train.
    """
    hidden_units = ([2048, 1024],
                    [1536, 768],
                    [1024, 512],
                    [2048, 1024, 512],
                    [1024, 512, 256])
    dirs = [str(i) for i in range(len(hidden_units))]
    errors = []

    for i, h_units in enumerate(hidden_units):
        regressor = tf.estimator.DNNRegressor(
            feature_columns=features_columns,
            hidden_units=h_units,
            model_dir=os.path.join("models", dirs[i])
        )

        epochs = 1000
        batch_size = 5
        total_steps = epochs * int(np.ceil(len([df_train]) / batch_size))
        regressor.train(
            input_fn=lambda: train_input_fn(df_train, batch_size=batch_size),
            steps=total_steps
        )

        y_pred = regressor.predict(input_fn=lambda: eval_input_fn(df_test, batch_size=batch_size))
        y_pred = np.array([value['predictions'][0] for value in iter(y_pred)])
        errors.append(_calculate_error(df_test["time"], y_pred))

    best_h_units_index = int(np.argmin(errors))
    return hidden_units[best_h_units_index]


def train_input_fn(df_train: pd.DataFrame, batch_size=5):
    df = df_train.copy()
    train_x, train_y = df, df.pop('time')
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_x), train_y))
    return dataset.batch(batch_size)


def eval_input_fn(df_test: pd.DataFrame, batch_size=5):
    df = df_test.copy()
    test_x, test_y = df, df.pop('time')
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(test_x), test_y))
    return dataset.batch(batch_size)


def _cast_to_dataframe(X: np.ndarray, y: np.ndarray):
    df = pd.DataFrame.from_dict({
        "x": X[:, 0],
        "y": X[:, 1],
        "time": y[:]
    })
    return df


def _calculate_error(y_true: np.ndarray, y_pred: np.ndarray):
    return sum((y_true - y_pred) ** 2)
