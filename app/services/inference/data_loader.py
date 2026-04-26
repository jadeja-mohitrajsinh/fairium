from io import BytesIO

import pandas as pd
from fastapi import UploadFile


def load_csv_from_upload(file: UploadFile) -> pd.DataFrame:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise ValueError("Please upload a CSV file.")

    raw = file.file.read()
    if not raw:
        raise ValueError("Uploaded file is empty.")

    try:
        dataframe = pd.read_csv(BytesIO(raw))
    finally:
        file.file.close()

    if dataframe.empty:
        raise ValueError("Uploaded dataset is empty.")

    return dataframe
