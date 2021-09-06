import pandas as pd
from os.path import splitext
from pyprind import ProgBar
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path: str):
    """
    Load data from .json or .csv file.

    Parameters
    ----------
    path : str
        file path to load.

    Returns
    -------
    data
        DataFrame object if file loaded correctly.
        None if file loading failed.
    """
    data = None
    try:
        _, extension = splitext(path)
        if extension == ".json":
            data = pd.read_json(path)
        elif extension == ".csv":
            data = pd.read_csv(path)
    except IOError:
        print("File loading failed")
    finally:
        return data


def validate_data(data: pd.DataFrame):
    """
    Process DataFrame object - dropping duplicates and checking data correctness.

    Parameters
    ---------
    data : DataFrame
        DataFrame object.

    Returns
    -------
    data
        processed and ready for analysis DataFrame object.
    """
    data.dropna(axis=0, inplace=True)
    data = data.drop_duplicates()
    data.reset_index(inplace=True)

    data["time"] = pd.to_datetime(data["time"].tolist(), format="%Y/%m/%d %H:%M:%S:%f")
    data["time"] = _map_dates_into_int(data["time"].values)
    data = data.sort_values(by=["time"])
    data["time"] = _preprocess_dates(data["time"].to_numpy())
    return data


def _map_dates_into_int(dates: list):
    return list(map(lambda date: np.datetime64(date).astype(np.int64), dates))


def _preprocess_dates(dates: np.ndarray):
    for i in range(len(dates)-1):
        dates[i] = (dates[i+1] - dates[i]) / 10 ** 9
    dates[-1] = dates[-2]
    dates = dates.astype(np.float32)
    return dates


def load_validate_data(path: str):
    """
    Load and validate data from file.
    Function performs the tasks of two
    functions - load_data() and validate_data().

    Parameters
    ----------
    path : str
        file path to load.

    Returns
    -------
    data
        loaded and validated data or None if something failed.
    """
    data = load_data(path)
    data = validate_data(data)
    print(data.head(5))
    return data


def prepare_train_data(df: pd.DataFrame,
                       feature: str = None,
                       feature_value=None):
    """
    Prepare useful data ready to train.

    Parameters
    ----------
    df : DataFrame
        Dataframe object which will be using to extract train data

    feature : str {default: None}
        Feature could be used for getting data for specific
        column's value in input DataFrame.

    feature_value : Any {default None}
        Specific column's value (see feature argument)

    Returns
    -------
    data : ndarray
        array of data ready to train in shape
        [(feature_0, feature_1, ...), <br />
         (feature_0, feature_1, ...), <br />
         ...]
    """
    assert df is not None, "Dataframe cannot be None"

    if feature in df.columns and feature_value is not None:
        indices = df[feature] == feature_value
    else:
        indices = np.ones(shape=(len(df),))

    Xs = df["x"][indices].values
    Ys = df["y"][indices].values
    Ts = df["time"][indices].values

    features_list = list(zip(Xs, Ys))

    sc = StandardScaler()
    features_list = sc.fit_transform(features_list)
    features_list = list(zip(features_list, Ts))
    features_list = list(map(lambda el: (el[0][0], el[0][1], el[1]), features_list))

    return np.array(features_list)


def create_csv_from_txt(path: str, out: str):
    """
    Create .csv file using data in .txt file.
    Process and validate all columns to save .csv file correctly.

    Parameters
    ----------
    path : str
        path to .txt file.

    out : str
        out file path.
    """
    try:
        with open(path, 'r') as in_file:
            lines_count = sum(1 for _ in in_file)

        pb = ProgBar(lines_count)

        with open(path, 'r') as in_file, open(out, 'w') as out_file:
            print("Converting .txt to .csv started")
            for i, line in enumerate(in_file):
                line = str(line).replace('\'$oid\': ', '')\
                    .replace('\'', '"').replace('"', '')\
                    .replace('{', '').replace('}', '')
                out_file.writelines(line)
                pb.update()
    except IOError:
        print('Reading .txt or out file file failed')
    finally:
        print("Converting from .txt ot .csv finished")


if __name__ == '__main__':
    create_csv_from_txt('data/tags_773k_ver01.txt', 'data/tags_773k_ver01.csv')
