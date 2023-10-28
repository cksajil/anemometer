import pandas as pd
from os.path import join
from micromlgen import port
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


DATA_FOLDER = "dataset"
DATA_FILE_NAME = "anemometer_data_full.csv"
HEADER_FILES_FOLDER = "headerfiles"
HEADER_FILE_NAME = "random_forest_model.h"
MODEL_PATH = "model"
MODEL_FILE = "best_random_forest_model.h5"

data = pd.read_csv(join(DATA_FOLDER, DATA_FILE_NAME), index_col=None, header=0)

y = data["speedlevel"]
X = data[["duration1", "duration2", "temperature", "humidity", "delta"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

y_pred = random_forest_model.predict(X_test)

val_mean_square_error = mean_squared_error(y_test, y_pred)
val_mean_absolute_error = mean_absolute_error(y_test, y_pred)
val_r2_score = r2_score(y_test, y_pred)

print("Validation mean square error of random forest model is: ", val_mean_square_error)
print("Validation mean absolute error of random forest model is: ", val_mean_absolute_error)
print("Validation R2 score of random forest model is: ", val_r2_score)


c_code = port(random_forest_model)
with open(join(HEADER_FILES_FOLDER, HEADER_FILE_NAME), "w") as f:
        f.write(c_code)
