import tensorflow as tf
from os.path import join
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
# from hex_converter import hex_to_c_array
from everywhereml.code_generators.tensorflow import tf_porter


# pip install everywhereml

EPOCHS = 500
indx = range(EPOCHS)
TFLITE_FOLDER = "tflite"
TFLITE_FILE_NAME = "dnn_model.tflite"

HEADER_FILES_FOLDER = "headerfiles"
HEADER_FILE_NAME = "dnn_model.h"



X, y = make_regression(n_samples=200, n_features=2, n_targets=1)



dnn_model = Sequential()


dnn_model.add(Dense(4, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
dnn_model.add(Dense(8, kernel_initializer='normal',activation='relu'))
dnn_model.add(Dense(1, kernel_initializer='normal',activation='linear'))


dnn_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
dnn_model.summary()

history = dnn_model.fit(X, y, epochs=EPOCHS, batch_size=32, validation_split = 0.2)

plt.plot(indx, history.history["loss"])
plt.plot(indx, history.history["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.legend(["train", "validation"], loc = "upper right")
plt.show()


# converter = tf.lite.TFLiteConverter.from_keras_model(dnn_model)
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# tflite_model = converter.convert()

# with open(join(TFLITE_FOLDER, TFLITE_FILE_NAME), "wb") as f:
#         f.write(tflite_model)

porter = tf_porter(dnn_model, X, y)
cpp_code = porter.to_cpp(instance_name='dnn_model', arena_size=4096)


with open(join(HEADER_FILES_FOLDER, HEADER_FILE_NAME), "w") as f:
        f.write(cpp_code)


print("Sample I/O Pairs")
print(X[4,:])
print(y[4])