from dotenv import load_dotenv

load_dotenv()

import os
import sys
import keras
import numpy as np
import pandas as pd


CLEAN_DATA_FOLDER = "clean_data"
RESULTS_DATA_FOLDER = "results"
WEIGHTS_FOLDER = os.path.join("models", "weights")
PATIENT_LIST = [2, 52, 54, 3, 24, 51, 34, 48, 25, 32, 43, 23, 59, 11, 49, 14, 33, 15]
SEED = 42

keras.utils.set_random_seed(SEED)


def main(model_name):
    model = keras.saving.load_model(os.path.join(WEIGHTS_FOLDER, f"{model_name}.keras"))
    ref_patient_df = pd.DataFrame({"Patient ID": PATIENT_LIST})
    X_test = np.load(os.path.join(CLEAN_DATA_FOLDER, "X_Test.npy"))
    Y_test = model.predict(X_test)
    Y_test[Y_test >= 0.5] = 1
    Y_test[Y_test < 0.5] = 0
    patient_df = pd.DataFrame(
        {"Patient ID": sorted(PATIENT_LIST), "Result": np.squeeze(Y_test).astype(int)}
    )
    patient_df = ref_patient_df.merge(patient_df, on="Patient ID", how="inner")
    patient_df.drop(columns=["Patient ID"]).to_csv(
        os.path.join(RESULTS_DATA_FOLDER, f"{model_name}_submission.csv"),
        index_label="index",
    )


if __name__ == "__main__":
    main(sys.argv[1])
