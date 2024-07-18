from utils import clean_dataframe, convert_df_to_array
import pandas as pd
import numpy as np
import os


STAGING_DATA_FOLDER = "staging_data"
CLEAN_DATA_FOLDER = "clean_data"


def main():
    # read the dataframes from staging folder
    train_readings_df = pd.read_csv(
        os.path.join(STAGING_DATA_FOLDER, "train_readings.csv")
    )
    train_patients_df = pd.read_csv(
        os.path.join(STAGING_DATA_FOLDER, "train_patients.csv")
    )
    test_readings_df = pd.read_csv(
        os.path.join(STAGING_DATA_FOLDER, "test_readings.csv")
    )
    test_patients_df = pd.read_csv(
        os.path.join(STAGING_DATA_FOLDER, "test_patients.csv")
    )
    # clean the readings data
    train_readings_df = clean_dataframe(train_readings_df)
    test_readings_df = clean_dataframe(test_readings_df)
    # convert the cleaned dataframes to a dataset
    train_array = convert_df_to_array(train_readings_df)
    test_array = convert_df_to_array(test_readings_df)
    # save the array
    np.save(os.path.join(CLEAN_DATA_FOLDER, "X_Train"), train_array)
    np.save(os.path.join(CLEAN_DATA_FOLDER, "X_Test"), test_array)
    np.save(
        os.path.join(CLEAN_DATA_FOLDER, "Y_Train"),
        train_patients_df.sort_values("Patient ID")["Result"].to_numpy(),
    )
    np.save(
        os.path.join(CLEAN_DATA_FOLDER, "Y_Test"),
        test_patients_df.sort_values("Patient ID")["Result"].to_numpy(),
    )


if __name__ == "__main__":
    main()
