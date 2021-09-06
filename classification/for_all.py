import numpy as np
from sklearn.preprocessing import StandardScaler
from plotting import *
from .k_nearest_neighbors import knn
from .decision_tree import decision_tree
from .random_forest import random_forest
from .svm import svm
from .logistic_regression import logistic_regression
from .multilayer_perceptron import mlp
from .team_classification import team
from .deep_neural_network import deep_neural_network


def classify(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Classify the input data using the best classification algorithm.

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
    cls
        trained classifier ready to predict.
    """
    team_ = team(X_train, y_train, X_test, y_test)
    return team_


def try_all_classifiers(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None,
                        compare: bool = True):
    """
    Try all classifiers which was implemented to classify input data.

    Parameters
    ----------
    X_train : ndarray
        Array of data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y_train : ndarray
        Array of labels belongs to X_train data.

    X_test : ndarray {default: None}
        Array of data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ]. Could be None.

    y_test : ndarray {default: None}
        Array of labels belongs to X_test data. Could be None.

    compare : bool {default: True}
        Are classifiers needed to be compared on some plots

    Returns
    -------
    clf
        The best trained classifier ready to predict.
    """
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    if X_test is not None:
        X_test = sc.transform(X_test)

    dnn_ = deep_neural_network(X_train, y_train, X_test, y_test)
    # plot_decision_regions(dnn_, X_train, y_train,
    #                       xlabel="x axis", ylabel="y axis", title="Deep Neural Network Classification result")

    team_ = team(X_train, y_train, X_test, y_test)
    # plot_decision_regions(team_, X_train, y_train,
    #                       xlabel="x axis", ylabel="y axis", title="Team Classification (Voting)")

    knn_ = knn(X_train, y_train, X_test, y_test)
    # plot_decision_regions(knn_, X_train, y_train,
    #                       xlabel="x axis", ylabel="y axis", title="KNN classification result")

    tree_ = decision_tree(X_train, y_train, X_test, y_test)
    # plot_decision_regions(tree_, X_train, y_train,
    #                       xlabel="x axis", ylabel="y axis", title="Decision Tree classification result")

    forest_ = random_forest(X_train, y_train, X_test, y_test)
    # plot_decision_regions(forest_, X_train, y_train,
    #                      xlabel="x axis", ylabel="y axis", title="Random Forest classification result")

    svm_ = svm(X_train, y_train, X_test, y_test)
    # plot_decision_regions(svm_, X_train, y_train,
    #                       xlabel="x axis", ylabel="y axis", title="SVM classification result")

    log_reg_ = logistic_regression(X_train, y_train, X_test, y_test)
    # plot_decision_regions(log_reg_, X_train, y_train,
    #                       xlabel="x axis", ylabel="y axis", title="Logistic Regression classification result")

    mlp_ = mlp(X_train, y_train, X_test, y_test)
    # plot_decision_regions(mlp_, X_train, y_train,
    #                       xlabel="x axis", ylabel="y axis", title="MultiLayer Perceptron classification result")

    if compare:
        classifiers = [knn_, tree_, forest_, log_reg_, svm_, mlp_, team_, dnn_]
        plot_classifiers_comparison(classifiers,
                                    X_train, y_train,
                                    xlabels=["x axis" for _ in range(len(classifiers))],
                                    ylabels=["y axis" for _ in range(len(classifiers))],
                                    titles=["KNN", "Decision Tree", "Random Forest", "Logistic Regression",
                                            "SVM", "MLP", "TEAM Classification", "Deep Neural Network"],
                                    suptitle="Classification Algorithms comparison")
    return team_
