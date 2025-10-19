import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

from ..generate_dataset.data_config import (
    ACTIONS,
    USED_LANDMARK_DIMENSION,
    VIDEO_FRAME_LENGTH,
)


log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)


def model_architecture():
    model = Sequential()
    model.add(
        LSTM(
            128,
            return_sequences=True,
            activation="relu",
            input_shape=(VIDEO_FRAME_LENGTH, USED_LANDMARK_DIMENSION),
        )
    )
    model.add(LSTM(256, return_sequences=True, activation="relu"))
    model.add(LSTM(128, return_sequences=False, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(ACTIONS.shape[0], activation="softmax"))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def __train_model(X_train, y_train):
    model = model_architecture()
    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
    return model


def __save_model(model, model_filename):
    model.save(model_filename)


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(pred, axis=1).tolist()
    print(multilabel_confusion_matrix(ytrue, yhat))
    accuracy = accuracy_score(ytrue, yhat)
    return accuracy


def train_model_workflow(X_train, y_train, X_test, y_test, model_filename):
    model = __train_model(X_train, y_train)
    __save_model(model, model_filename)
    evaluate_model(model, X_test, y_test)
    return model
