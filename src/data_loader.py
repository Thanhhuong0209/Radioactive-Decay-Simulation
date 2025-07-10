import pandas as pd
import os
# Add Dask import
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

import streamlit as st
import gdown

# Danh sách các file Google Drive cần tải
GDRIVE_FILES = [
    {"file_id": "13RsFtygqgl-SyE46OE3BCrmMwwOtV4bO", "local_path": "data/file1.csv"},
    {"file_id": "15ngWIZvVr3-OrzTFmJ7hqt7HqVElcSNm", "local_path": "data/file2.csv"},
    {"file_id": "1ZQKjQHTJmoR53G1Jon1psxSP4AdHI7Zx", "local_path": "data/file3.csv"},
    {"file_id": "1QsQU9EBMusa3QvTODUFIGCr9BVmF5DrO", "local_path": "data/file4.csv"},
    {"file_id": "1B8_L9Kxz4UWNluIP0tdaiNUiM1dGQt7u", "local_path": "data/file5.csv"},
]

def download_gdrive_files():
    for file in GDRIVE_FILES:
        file_path = file["local_path"]
        file_id = file["file_id"]
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            print(f"Downloading {file_path} from Google Drive...")
            gdown.download(url, file_path, quiet=False)
        else:
            print(f"File {file_path} already exists. Skipping download.")

# Gọi hàm này ở đầu chương trình hoặc trước khi load dữ liệu
download_gdrive_files()

def load_opr_measurements(filepath="data/measurements.csv", nrows=None):
    """
    Read OpenRadiation data (measurements.csv) with actual columns, separated by semicolons.
    - Only select columns: value, latitude, longitude, altitude, startTime
    - Rename: value -> radiation, startTime -> date
    - Return cleaned DataFrame
    - If the dataset is large, automatically use Dask for speedup
    """
    usecols = ["value", "latitude", "longitude", "altitude", "startTime"]
    parse_dates = ["startTime"]
    # Use Dask if nrows is large or file is large
    use_dask = False
    if nrows is not None and nrows > 100000 and DASK_AVAILABLE:
        use_dask = True
    elif os.path.exists(filepath) and DASK_AVAILABLE:
        file_size = os.path.getsize(filepath)
        if file_size > 50*1024*1024:  # >50MB
            use_dask = True
    if use_dask:
        st.warning("Using Dask to speed up large data processing!")
        df = dd.read_csv(filepath, usecols=usecols, parse_dates=parse_dates, sep=';', dtype={'altitude': 'float64'})
        if nrows:
            df = df.head(nrows, compute=True)
        else:
            df = df.compute()
        df = pd.DataFrame(df)  # Ensure return as pandas DataFrame
    else:
        if nrows:
            df = pd.read_csv(filepath, usecols=usecols, parse_dates=parse_dates, sep=';', nrows=nrows)
        else:
            df = pd.read_csv(filepath, usecols=usecols, parse_dates=parse_dates, sep=';')
    df = df.rename(columns={
        "value": "radiation",
        "latitude": "latitude",
        "longitude": "longitude",
        "startTime": "date"
    })
    df = df[df["radiation"] > 0]
    return df

def load_chernobyl(filepath="data/Chernobyl_ Chemical_Radiation.csv"):
    """
    Read Chernobyl data (Chernobyl_Chemical_Radiation.csv).
    - Normalize time, location, and measurement columns.
    - Return cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    # Assume columns: Date, Location, Radiation, Unit
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Radiation" in df.columns:
        df = df[df["Radiation"] > 0]
    return df

def load_chernobyl_air(filepath="data/chernobyl_air_concentration.csv"):
    """
    Read Chernobyl air concentration data (chernobyl_air_concentration.csv).
    - Normalize time, location, and measurement columns.
    - Return cleaned DataFrame.
    """
    df = pd.read_csv(filepath)
    # Assume columns: Date, Location, Value, Unit
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Value" in df.columns:
        df = df[df["Value"] > 0]
    return df 