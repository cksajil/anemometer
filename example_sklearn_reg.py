# https://eloquentarduino.com/libraries/micromlgen/

from os.path import join
from micromlgen import port
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HEADER_FILES_FOLDER = "headerfiles"
FILE_NAME = "linear_regression_regr.h"

X, y = datasets.load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

regr = LinearRegression()


regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("R2 score: %.2f" % r2_score(y_test, y_pred))


c_code = port(regr)

with open(join(HEADER_FILES_FOLDER, FILE_NAME), "w") as f:
    f.write(c_code)
