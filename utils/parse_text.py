import pandas as pd


def correct_time(data):
    new_data = []
    new_data.append(list(data[0]))
    new_data[0][0] = f"00:{data[0][0]}"
    hour_change = False
    for i in range(1, len(data)):
        prev_min = int(data[i - 1][0].split(":")[0])
        curr_min = int(data[i][0].split(":")[0])
        if curr_min < prev_min:
            hour_change = True
        new_data.append(list(data[i]))
        if hour_change:
            new_data[i][0] = f"01:{data[i][0]}"
        else:
            new_data[i][0] = f"00:{data[i][0]}"
    return new_data


def text_to_readings_df(path, test):
    patient_id = None
    header = None
    data = None
    with open(path, "r") as fp:
        patient_id = int(fp.readline().strip().split(":")[1])
        if not test:
            _ = fp.readline()
        _ = fp.readline()
        header = fp.readline().strip().split()
        data = [
            [float(y) if idx != 0 else y for idx, y in enumerate(x.strip().split())]
            for x in fp.readlines()
        ]
    data = correct_time(data)
    data_df = pd.DataFrame(data, columns=header)
    data_df["Patient ID"] = patient_id
    data_df = data_df.rename(columns={"Min:Sec": "Time"})
    data_df["Time"] = pd.to_timedelta(data_df["Time"]).dt.total_seconds()
    return data_df


def text_to_patients_df(path, test):
    patient_id = None
    result = None
    with open(path, "r") as fp:
        patient_id = int(fp.readline().strip().split(":")[1])
        if not test:
            result = fp.readline().strip().split(":")[1].strip().lower()
            if result == "positive":
                result = 1
            else:
                result = 0
    return pd.DataFrame({"patient_id": [patient_id], "result": [result]})
