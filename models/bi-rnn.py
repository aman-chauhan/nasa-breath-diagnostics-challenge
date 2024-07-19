from dotenv import load_dotenv

load_dotenv()

import os
import keras
import numpy as np


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
        batch_size=5,
        epochs=8,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.CSVLogger(
                os.path.join(RESULT_DATA_FOLDER, f"{MODEL_NAME}.csv")
            ),
        ],
    )
    model.save(os.path.join(WEIGHTS_FOLDER, f"{MODEL_NAME}.keras"))
