from data_management import *
from clustering import *
from classification import *
from regression import *
from os.path import join
from sklearn.model_selection import train_test_split


def main():
    path_csv = join('data', 'tags_773k_ver01.csv')
    path_json = join('data', 'tags_08_07_21_ver02.json')

    data = load_validate_data(path_csv)
    unique_column = "ID"

    for id in np.unique(data[unique_column]):
        X = prepare_train_data(data, unique_column, id)
        points = X[:, [0, 1]]

        # y = clusterize(points)
        y = try_all_clusterizers(points)

        X_train, X_test, y_train, y_test = train_test_split(points, y, test_size=0.20, stratify=y, random_state=1)
        # clf = classify(X_train, y_train, X_test, y_test)
        clf = try_all_classifiers(X_train, y_train, X_test, y_test)

        for group in np.unique(y):
            indices = y == group

            group_points = X[indices][:, [0, 1]]
            group_points_times = X[indices][:, 2]

            X_train, X_test, y_train, y_test = train_test_split(group_points, group_points_times, test_size=0.20, random_state=1)
            # reg = regression(X_train, y_train, X_test, y_test)
            reg = try_all_regressors(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
