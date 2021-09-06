
# functions that use all regressors
from .for_all import try_all_regressors
from .for_all import regression

# functions that use each of regressors
from .decision_tree import decision_tree
from .k_nearest_neighbors import k_nearest_neighbors
from .neural_network import neural_network
from .svm import svm_regression
