# Source https://eloquentarduino.com/tensorflow-lite-esp32/
# pip install everywhereml
# Install EloquentTinyML Library in Arduino

import pandas as pd
from os.path import join
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from everywhereml.code_generators.tensorflow import tf_porter
from tensorflow.keras.optimizers import Adam


DATA_FOLDER = "dataset"
DATA_FILE_NAME = "anemometer_data_full.csv"
HEADER_FILES_FOLDER = "headerfiles"
HEADER_FILE_NAME = "dnn_model.h"

EPOCHS = 50
BATCH_SIZE = 32
indx = range(EPOCHS)
data = pd.read_csv(join(DATA_FOLDER, DATA_FILE_NAME), index_col=None, header=0)

y = data["speedlevel"]
X = data[["duration1", "duration2", "temperature", "humidity", "delta"]]

print("Shape of total data is")
print(X.shape)
print(y.shape)
print(X.head(2))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

dnn_model = Sequential()
dnn_model.add(Dense(4, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
dnn_model.add(Dense(8, kernel_initializer='normal',activation='relu'))
dnn_model.add(Dense(16, kernel_initializer='normal',activation='relu'))
dnn_model.add(Dense(8, kernel_initializer='normal',activation='relu'))
dnn_model.add(Dense(1, kernel_initializer='normal',activation='linear'))


dnn_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
dnn_model.summary()

history = dnn_model.fit(X_train, 
                        y_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data = (X_test, y_test))

plt.plot(indx, history.history["loss"])
plt.plot(indx, history.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.legend(["train", "validation"], loc = "upper right")
plt.show()

porter = tf_porter(dnn_model, X, y)
cpp_code = porter.to_cpp(instance_name='dnn_model', arena_size=4096)


with open(join(HEADER_FILES_FOLDER, HEADER_FILE_NAME), "w") as f:
        f.write(cpp_code)
