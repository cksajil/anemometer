# https://eloquentarduino.com/libraries/micromlgen/

from os.path import join
from micromlgen import port
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

HEADER_FILES_FOLDER = "headerfiles"
FILE_NAME = "random_forest_clf.h"

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    clf_rf = RandomForestClassifier(n_estimators=10).fit(X, y)
    c_code = port(clf_rf)

    with open(join(HEADER_FILES_FOLDER, FILE_NAME), "w") as f:
        f.write(c_code)
