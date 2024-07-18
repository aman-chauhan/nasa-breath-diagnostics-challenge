from dotenv import load_dotenv

load_dotenv()

import os
import keras
import numpy as np


CLEAN_DATA_FOLDER = "clean_data"
RESULT_DATA_FOLDER = "results"
CHECKPOINT_FOLDER = os.path.join("models", "checkpoints")
WEIGHTS_FOLDER = os.path.join("models", "weights")
SEED = 42

keras.utils.set_random_seed(SEED)


def create_rnn_model():
    i = keras.Input(shape=(None, 64), name="rnn_input")
    r = keras.layers.SimpleRNN(
        128,
        kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),
        seed=SEED,
        name="rnn_layer",
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
        name="rnn_output",
    )(l)
    return keras.Model(inputs=i, outputs=o, name="rnn")


if __name__ == "__main__":
    model = create_rnn_model()
    model.compile(
        optimizer="adam",
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
    history = model.fit(
        x=X_train,
        y=Y_train,
        batch_size=15,
        epochs=100,
        validation_split=0.33333,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20),
            keras.callbacks.CSVLogger(os.path.join(RESULT_DATA_FOLDER, "rnn.csv")),
        ],
    )
    model.save(os.path.join(WEIGHTS_FOLDER, "rnn.keras"))
