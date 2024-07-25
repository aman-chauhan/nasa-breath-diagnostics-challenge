from dotenv import load_dotenv

load_dotenv()

import os
import keras
import numpy as np
from utils.data_loader import DataLoader


CLEAN_DATA_FOLDER = "clean_data"
RESULT_DATA_FOLDER = "results"
WEIGHTS_FOLDER = os.path.join("models", "weights")
MODEL_NAME = "bi-rnn"
SEED = 42

keras.utils.set_random_seed(SEED)


def create_bi_rnn_model():
    i = keras.Input(shape=(None, 64), name=f"{MODEL_NAME}_input")
    r = keras.layers.Bidirectional(
        keras.layers.SimpleRNN(
            128,
            kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),
            seed=SEED,
            name=f"{MODEL_NAME}_layer",
        )
    )(i)
    l = keras.layers.Dense(
        256,
        activation="relu",
        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),
        name="dense_layer",
    )(r)
    o = keras.layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),
        name=f"{MODEL_NAME}_output",
    )(l)
    return keras.Model(inputs=i, outputs=o, name=f"{MODEL_NAME}")


if __name__ == "__main__":
    model = create_bi_rnn_model()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.AUC(),
            keras.metrics.F1Score(average="micro", threshold=0.5),
        ],
    )
    model.summary()
    X_train = np.load(os.path.join(CLEAN_DATA_FOLDER, "X_Train.npy"))
    Y_train = np.load(os.path.join(CLEAN_DATA_FOLDER, "Y_Train.npy"))
    Y_train = np.expand_dims(Y_train, -1)
    X_valid = X_train[-9:]
    Y_valid = Y_train[-9:]
    X_train = X_train[:-9]
    Y_train = Y_train[:-9]
    history = model.fit(
        x=DataLoader(X_train, Y_train, batch_size=9, augment=True),
        epochs=1000,
        validation_data=DataLoader(X_valid, Y_valid, batch_size=9, augment=False),
        shuffle=False,
        callbacks=[
            keras.callbacks.CSVLogger(
                os.path.join(RESULT_DATA_FOLDER, f"{MODEL_NAME}.csv")
            ),
            keras.callbacks.ModelCheckpoint(
                os.path.join(WEIGHTS_FOLDER, f"{MODEL_NAME}.keras"),
                monitor="val_loss",
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=100),
        ],
    )
