#!/usr/bin/env python3
import datetime
import numpy as np
import sys
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
    pd.set_option("mode.copy_on_write", True)
from os.path import expanduser, realpath

sys.path.append(realpath(expanduser("../../../diagnostic-ultrasound-centre-project")))

from data_directories import data_directories
from src.utils.display import printdf
from src.utils.standardise_column_names import standardise_column_names
from src.utils.reorder_pandas_columns import reorder_pandas_columns

# Define constants
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72
RANDOM_SAMPLE_SIZE = 13
RANDOM_SEED = 42

# Import raw data
raw_data = data_directories["raw_data_dir"] + "/000-real-ultrasound-patronage-data.xlsx"

df_raw_01 = pd.read_excel(raw_data, sheet_name="Patronage_box_data")
df_raw_02 = pd.read_excel(raw_data, sheet_name="Patronage_flat_data")

printdf(df_raw_01)
printdf(df_raw_02)

df_raw_01.describe().T
df_raw_02.describe().T

df_standardised_01 = standardise_column_names(df_raw_01)
df_standardised_02 = standardise_column_names(df_raw_02)

printdf(df_standardised_01)
printdf(df_standardised_02)

remap_month_name_to_number = {
    "Jan.": 1,
    "Feb.": 2,
    "Mar.": 3,
    "Apr.": 4,
    "May": 5,
    "Jun.": 6,
    "Jul.": 7,
    "Aug.": 8,
    "Sep.": 9,
    "Oct.": 10,
    "Nov.": 11,
    "Dec.": 12,
}

df_standardised_01["month_number"] = df_standardised_01["month"].map(remap_month_name_to_number)

df_standardised_01["date_str"] = df_standardised_01["year"].astype(str) + "-" + df_standardised_01["month"].astype(str)

# How to make pd.to_datetime insert last day of month instead of first when input is limited to 'yyyy-mm'? -> https://stackoverflow.com/a/59190866
try:
    df_standardised_01["datetime"] = pd.to_datetime(df_standardised_01["date_str"]) + pd.offsets.MonthEnd()
except ValueError:
    df_standardised_01["datetime"] = pd.to_datetime(
        dict(
            year = df_standardised_01["year"],
            month = df_standardised_01["month_number"],
            day = 28,
        )
    )

df_standardised_01.drop(["year", "month", "month_number", "date_str"], axis=1, inplace=True)

df_standardised_01["year"] = df_standardised_01["datetime"].dt.year

df_standardised_01["month"] = df_standardised_01["datetime"].dt.month

# Create Year-Month Column from Dates -> https://dfrieds.com/data-analysis/create-year-month-column.html
df_standardised_01["year_month"] = df_standardised_01["datetime"].dt.strftime("%Y-%m")


df_standardised = df_standardised_01[["datetime", "year_month", "year", "month", "obstetrics", "pelvic", "abdominal", "transrectal", "breast", "transvaginal", "folliculometry", "thyroid_neck", "scrotal", "doppler", "anomaly", "ocular", "musculoskeletal", "other_special_scans", "echocardiography"]]

path_for_csv = data_directories["interim_data_dir"] + "/df_standardised.csv"

df_standardised.to_csv(path_for_csv, index=False)

printdf(df_standardised)

