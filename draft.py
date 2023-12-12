"""
    ??  数据长啥样
    ??  图上的那些参数什么意思
    ??  decision_scores_ 是什么
"""

import numpy as np
from adbench.baseline.PyOD import PYOD
from pyod.models.lof import LOF
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize


if __name__ == "__main__":
    """
    Prepare the data
    """
    # # 01 Generate sample data
    # contamination = 0.1  # percentage of outliers
    # n_train = 200
    # n_test = 100

    # X_train, y_train, X_test, y_test = generate_data(
    #     n_train=n_train,
    #     n_test=n_test,
    #     n_features=2,
    #     contamination=contamination,
    #     random_state=42,
    # )

    # 02 Load sample data
    data = np.load("Datasets/13_fraud.npz", allow_pickle=True)  # (284807, 29)
    # data = np.load("Datasets/15_Hepatitis.npz", allow_pickle=True)  # (80, 19)
    # data = np.load("Datasets/16_http.npz", allow_pickle=True)  # (567498, 3)

    X, y = data["X"], data["y"]
    print("X.shape, y.shape", X.shape, y.shape)
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train + 1 :], y[n_train + 1 :]

    """
    Fit the model
    """
    clf = PYOD(seed=42, model_name="LOF")
    clf_name = "LOF"
    clf = LOF()
    # clf = PYOD(seed=42, model_name="ECOD")
    clf_name = "ECOD"
    clf = ECOD()
    # clf = PYOD(seed=42, model_name="KNN")
    # clf_name = "KNN"
    # clf = KNN()
    clf.fit(X_train)

    """
    Validate the model
    """
    # training data
    y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_
    # test data
    y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)

    """
    Output and Visualize
    """
    print("\nOn Training Data:")
    evaluate_print(clf, y_train, y_train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf, y_test, y_test_scores)

    # # 只适合 n_features=2 的数据
    # visualize(
    #     clf,
    #     X_train,
    #     y_train,
    #     X_test,
    #     y_test,
    #     y_train_pred,
    #     y_test_pred,
    #     show_figure=True,
    #     save_figure=False,
    # )


# >>> conda activate ADBench
