
# functions that use all classifiers
from .for_all import try_all_classifiers, classify

# functions for each classifiers
from .decision_tree import decision_tree
from .deep_neural_network import deep_neural_network
from .k_nearest_neighbors import knn
from .logistic_regression import logistic_regression
from .multilayer_perceptron import mlp
from .random_forest import random_forest
from .svm import svm
from .team_classification import team
