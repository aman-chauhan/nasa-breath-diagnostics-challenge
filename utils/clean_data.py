import pandas as pd


def clean_dataframe(df: pd.DataFrame):
    # truncate readings that went on beyond 14 minutes
    df = df[df["Time"] < 841.0]
    # get the mean readings for the first 5 minutes for each patient
    patient_means = (
        df[df["Time"] < 301]
        .groupby("Patient ID")
        .mean()
        .reset_index()
        .drop(columns="Time")
    )
    # merge the means with the original dataframe
    result = df.merge(patient_means, on="Patient ID", suffixes=("_sample", "_mean"))
    # for each column subtract the means for each patient
    for col in [f"D{i+1}" for i in range(64)]:
        result[col] = result[f"{col}_sample"] - result[f"{col}_mean"]
        result.drop(columns=[f"{col}_sample", f"{col}_mean"], inplace=True)
    return result


def convert_df_to_array(df: pd.DataFrame):
    # sort the dataframe by Patient ID and Time
    df = df.sort_values(["Patient ID", "Time"])
    # total number of patients
    patients = sorted(df["Patient ID"].unique().tolist())
    # number of readings for each patient
    num_readings = len(df[df["Patient ID"] == patients[0]])
    # drop the Patient ID and Time columns
    cols = df.columns.difference(["Patient ID", "Time"])
    df_array = df[cols].to_numpy()
    # reshape the array
    df_array = df_array.reshape(len(patients), num_readings, len(cols))
    return df_array
