# Source https://eloquentarduino.com/tensorflow-lite-esp32/
# pip install everywhereml
# Install EloquentTinyML Library in Arduino

import pandas as pd
from os.path import join
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from everywhereml.code_generators.tensorflow import tf_porter


DATA_FOLDER = "dataset"
DATA_FILE_NAME = "anemometer_data_full.csv"
HEADER_FILES_FOLDER = "headerfiles"
HEADER_FILE_NAME = "dnn_model.h"
PLOT_PATH = "plots"
FIG_NAME = "train_validation_curve.png"
MODEL_PATH = "model"
MODEL_FILE = "best_model_{epoch:03d}_{val_mean_absolute_error:03f}.h5"

EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 0.001
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

adamopt = Adam(learning_rate = LEARNING_RATE)

checkpoint_callback = ModelCheckpoint(
        filepath=join(MODEL_PATH, MODEL_FILE),
        save_weights_only=True,
        monitor="val_mean_absolute_error",
        mode="min",
        save_best_only=True)

dnn_model = Sequential()
dnn_model.add(Dense(32, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
dnn_model.add(Dense(64, kernel_initializer='normal',activation='relu'))
dnn_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
dnn_model.add(Dense(16, kernel_initializer='normal',activation='relu'))
dnn_model.add(Dense(1, kernel_initializer='normal',activation='linear'))


dnn_model.compile(loss='mean_absolute_error', 
                  optimizer=adamopt,
                  metrics=['mean_absolute_error'])

dnn_model.summary()

history = dnn_model.fit(X_train, 
                        y_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data = (X_test, y_test),
                        callbacks=[checkpoint_callback])

plt.plot(indx, history.history["loss"])
plt.plot(indx, history.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.legend(["train", "validation"], loc = "upper right")
plt.savefig(join(PLOT_PATH, FIG_NAME), dpi=200)
plt.show()

porter = tf_porter(dnn_model, X, y)
cpp_code = porter.to_cpp(instance_name='dnn_model', arena_size=4096)


with open(join(HEADER_FILES_FOLDER, HEADER_FILE_NAME), "w") as f:
        f.write(cpp_code)
