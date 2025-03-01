from zipfile import ZipFile
from io import BytesIO, StringIO
import urllib.request as urllib2
import pandas as pd
from torch_geometric.data import TemporalData
from sklearn import preprocessing
import torch
import numpy as np


def get_cbs_data(path="./data/cbs/202207-capitalbikeshare-tripdata.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if path is None:
        # download data
        cbs_url = "https://s3.amazonaws.com/capitalbikeshare-data/202207-capitalbikeshare-tripdata.zip"
        r = urllib2.urlopen(cbs_url).read()
        file = ZipFile(BytesIO(r))
        cbs_b = file.read("202207-capitalbikeshare-tripdata.csv")
        cbs0 = pd.read_csv(StringIO(str(cbs_b, "utf-8")))
    else:
        try:
            cbs0 = pd.read_csv(path)
        except KeyError:
            # download data
            cbs_url = "https://s3.amazonaws.com/capitalbikeshare-data/202207-capitalbikeshare-tripdata.zip"
            r = urllib2.urlopen(cbs_url).read()
            file = ZipFile(BytesIO(r))
            cbs_csv = file.open(path)
            cbs0 = pd.read_csv(path)

    cbs = cbs0

    # preprocess
    # compute duration
    cbs["started_at"] = pd.to_datetime(cbs["started_at"])
    cbs["ended_at"] = pd.to_datetime(cbs["ended_at"])
    cbs["duration_s"] = (cbs["ended_at"] - cbs["started_at"]).dt.seconds
    cbs["log_duration"] = np.log(cbs["duration_s"] + 1e-1)

    # drop loops
    cbs = cbs[cbs["start_station_id"] != cbs["end_station_id"]]

    # rescale start time
    min_started_at = cbs["started_at"].min()
    cbs["start_time"] = (cbs["started_at"] - min_started_at).dt.seconds

    # select columns
    columns = [
        "start_station_id",
        "end_station_id",
        "start_time",
        "log_duration",
        "rideable_type",
        "member_casual",
    ]
    df = cbs[columns].dropna().reset_index(drop=True)
    df = df.astype({"start_station_id": int, "end_station_id": int})

    msg_columns = ["log_duration", "rideable_type", "member_casual"]
    categorical_cols = ["rideable_type", "member_casual"]

    col2lenc = {}
    col2K = {}
    for i, col in enumerate(msg_columns):
        if col in categorical_cols:
            le = preprocessing.LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])
            col2lenc[i] = le
            col2K[i] = len(le.classes_)
        else:
            col2K[i] = "gmm"

    # relabel node ids
    unique_station_ids = list(
        set(
            df["start_station_id"].unique().tolist()
            + df["end_station_id"].unique().tolist()
        )
    )
    le = preprocessing.LabelEncoder()
    le.fit(unique_station_ids)
    df["start_station_id"] = le.transform(df["start_station_id"])
    df["end_station_id"] = le.transform(df["end_station_id"])
    col2lenc["start_station_id"] = le
    col2lenc["end_station_id"] = le

    # sort by start time
    df = df.sort_values(by="start_time").reset_index(drop=True)

    data = TemporalData(
        src=torch.tensor(df["start_station_id"].to_numpy()).long().to(device),
        dst=torch.tensor(df["end_station_id"].to_numpy()).long().to(device),
        t=torch.tensor(df["start_time"].to_numpy()).long().to(device),
        msg=torch.tensor(df[msg_columns].to_numpy()).float().to(device),
    )

    return data.to(device), col2K
