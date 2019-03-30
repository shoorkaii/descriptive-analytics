import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import floor
from sklearn import metrics
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

kc_house_data = pd.read_csv('data/kc_house_data.csv')
kc_house_data_count_row = kc_house_data.shape[0]

feature_col_names = ["sqft_living15", "sqft_lot", "grade", "sqft_above", "yr_built", "bedrooms", "floors"
    , "bathrooms", "zipcode"]
predicted_col_name = ['price']

X = kc_house_data[feature_col_names].values
y = kc_house_data[predicted_col_name].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

total_accuracy = {}
total_mae = {}


def print_result(model, model_name):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)
    print(model_name)
    print("# exactly matched labels: ", floor(accuracy * kc_house_data_count_row * .25))
    print("mean absolute error: ", mae)
    total_accuracy[str((str(model).split('(')[0]))] = accuracy
    total_mae[str((str(model).split('(')[0]))] = mae


def plot_total_accuracy():
    data = total_accuracy.values()
    labels = total_accuracy.keys()
    plt.plot([i for i, e in enumerate(data)], data, 'mo', markersize=12)
    plt.xticks([i for i, e in enumerate(labels)], [l[0:16] for l in labels])
    plt.title("Model Vs Accuracy", fontsize=14)
    plt.xlabel('Model', fontsize=13)
    plt.xticks(rotation=10)
    plt.ylabel('Accuracy', fontsize=13)
    plt.show()


def plot_total_mae():
    data = total_mae.values()
    labels = total_mae.keys()
    plt.plot([i for i, e in enumerate(data)], data, 'mo', markersize=12)
    plt.xticks([i for i, e in enumerate(labels)], [l[0:16] for l in labels])
    plt.title("Model Vs MAE", fontsize=14)
    plt.xlabel('Model', fontsize=13)
    plt.xticks(rotation=10)
    plt.ylabel('MAE', fontsize=13)
    plt.show()


def linear_svm_classifier():
    _svm = SVC(kernel="linear", C=0.025)
    _svm.fit(X_train, y_train.ravel())
    print_result(_svm, linear_svm_classifier.__name__)


def rbf_svm_classifier():
    _svm = SVC(gamma=2, C=1)
    _svm.fit(X_train, y_train.ravel())
    print_result(_svm, rbf_svm_classifier.__name__)


def decision_tree_classifier():
    _dtc = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
    _dtc.fit(X_train, y_train.ravel())
    print_result(_dtc, decision_tree_classifier.__name__)
    # feature_importances_plot(_dtc)


def feature_importances_plot(_dtc):
    feature_importances = pd.Series(_dtc.feature_importances_, index=feature_col_names)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title("Feature Importance:")
    plt.show()


def ada_boost_classifier():
    _ada_boost = AdaBoostClassifier()
    _ada_boost.fit(X_train, y_train.ravel())
    print_result(_ada_boost, ada_boost_classifier.__name__)


def random_forest_classifier():
    _random_forest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    _random_forest.fit(X_train, y_train.ravel())
    print_result(_random_forest, random_forest_classifier.__name__)


def knn_classifier():
    _knn = KNeighborsClassifier(n_neighbors=3)
    _knn.fit(X_train, y_train.ravel())
    print_result(_knn, knn_classifier.__name__)


def gaussian_process_classifier():
    kernel = 1.0 * RBF(1.0)
    _gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
    _gpc.fit(X_train, y_train.ravel())
    print_result(_gpc, gaussian_process_classifier.__name__)


def gnb_classifier():
    _gnb = GaussianNB()
    _gnb.fit(X_train, y_train.ravel())
    print_result(_gnb, gnb_classifier.__name__)


def mlp_classifier():
    _mlp = MLPClassifier(alpha=1)
    _mlp.fit(X_train, y_train.ravel())
    print_result(_mlp, mlp_classifier.__name__)


def main():
    decision_tree_classifier()
    ada_boost_classifier()
    random_forest_classifier()
    knn_classifier()
    gnb_classifier()
    mlp_classifier()
    # gaussian_process_classifier()
    # rbf_svm_classifier()
    # linear_svm_classifier()
    plot_total_accuracy()
    plot_total_mae()


if __name__ == "__main__":
    main()
