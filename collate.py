from utils.parse_text import *
import pandas as pd
import os


RAW_DATA_FOLDER = "raw_data"
STAGING_DATA_FOLDER = "staging_data"


def collate(folder_path, test):
    collate_reading_df = None
    collate_patient_df = None
    for _, _, files in os.walk(folder_path):
        for filename in files:
            filepath = os.path.join(folder_path, filename)
            reading_df = text_to_readings_df(filepath, test)
            patient_df = text_to_patients_df(filepath, test)
            if collate_reading_df is None:
                collate_reading_df = reading_df
            else:
                collate_reading_df = pd.concat(
                    [collate_reading_df, reading_df], ignore_index=True
                )
            if collate_patient_df is None:
                collate_patient_df = patient_df
            else:
                collate_patient_df = pd.concat(
                    [collate_patient_df, patient_df], ignore_index=True
                )
    return collate_reading_df, collate_patient_df


def main():
    train_folder = os.path.join(RAW_DATA_FOLDER, "train")
    train_reading_df, train_patient_df = collate(train_folder, False)
    train_reading_df.to_csv(
        os.path.join(STAGING_DATA_FOLDER, "train_readings.csv"), index=False
    )
    train_patient_df.to_csv(
        os.path.join(STAGING_DATA_FOLDER, "train_patients.csv"), index=False
    )
    test_folder = os.path.join(RAW_DATA_FOLDER, "test")
    test_reading_df, test_patient_df = collate(test_folder, True)
    test_reading_df.to_csv(
        os.path.join(STAGING_DATA_FOLDER, "test_readings.csv"), index=False
    )
    test_patient_df.to_csv(
        os.path.join(STAGING_DATA_FOLDER, "test_patients.csv"), index=False
    )


if __name__ == "__main__":
    main()
