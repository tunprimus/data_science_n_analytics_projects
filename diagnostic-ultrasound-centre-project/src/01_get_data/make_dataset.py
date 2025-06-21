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
from pathlib import Path

proj_dir_path = Path.cwd().parent.parent
proj_dir_name = Path.cwd().parent.parent.name
sys.path.append(realpath(expanduser(proj_dir_path)))

print(f"Project directory name is: {proj_dir_name}")

from data_directories import data_directories
from src.utils.display import printdf
from src.utils.standardise_column_names import standardise_column_names
from src.utils.reorder_pandas_columns import reorder_pandas_columns
from src.utils.dynamic_import_spreadsheet_into_sqlite import create_table, insert_into_db, process_csv_into_sqlite, process_spreadsheet_into_sqlite

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
raw_data02 = data_directories["raw_data_dir"] + "/real-ultrasound-patronage-data.xlsx"

df_raw_01 = pd.read_excel(raw_data, sheet_name="Patronage_box_data")
df_raw_02 = pd.read_excel(raw_data, sheet_name="Patronage_flat_data")

print(type(df_raw_01))
print(type(df_raw_01.columns))

# printdf(df_raw_01)
# printdf(df_raw_02)

df_raw_01.describe().T
df_raw_02.describe().T

df_standardised_01 = standardise_column_names(df_raw_01)
df_standardised_02 = standardise_column_names(df_raw_02)

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

df_standardised_02["month_number"] = df_standardised_02["month"].map(remap_month_name_to_number)

df_standardised_02["date_str"] = df_standardised_02["year"].astype(str) + "-" + df_standardised_02["month"].astype(str)

# How to make pd.to_datetime insert last day of month instead of first when input is limited to 'yyyy-mm'? -> https://stackoverflow.com/a/59190866
try:
    df_standardised_01["datetime"] = pd.to_datetime(df_standardised_01["date_str"]) + pd.offsets.MonthEnd()

    df_standardised_02["datetime"] = pd.to_datetime(df_standardised_02["date_str"]) + pd.offsets.MonthEnd()
except ValueError:
    df_standardised_01["datetime"] = pd.to_datetime(
        dict(
            year = df_standardised_01["year"],
            month = df_standardised_01["month_number"],
            day = 28,
        )
    )

    df_standardised_02["datetime"] = pd.to_datetime(
        dict(
            year = df_standardised_02["year"],
            month = df_standardised_02["month_number"],
            day = 28,
        )
    )

df_standardised_01.drop(["year", "month", "month_number", "date_str"], axis=1, inplace=True)

df_standardised_01["year"] = df_standardised_01["datetime"].dt.year

df_standardised_01["month"] = df_standardised_01["datetime"].dt.month

df_standardised_02.drop(["year", "month", "month_number", "date_str"], axis=1, inplace=True)

df_standardised_02["year"] = df_standardised_02["datetime"].dt.year

df_standardised_02["month"] = df_standardised_02["datetime"].dt.month

# Create Year-Month Column from Dates -> https://dfrieds.com/data-analysis/create-year-month-column.html
df_standardised_01["year_month"] = df_standardised_01["datetime"].dt.strftime("%Y-%m")

df_standardised_02["year_month"] = df_standardised_02["datetime"].dt.strftime("%Y-%m")

# Re-order the columns
df_wide = df_standardised_01[["datetime", "year_month", "year", "month", "obstetrics", "pelvic", "abdominal", "transrectal", "breast", "transvaginal", "folliculometry", "thyroid_neck", "scrotal", "doppler", "anomaly", "ocular", "musculoskeletal", "other_special_scans", "echocardiography"]]

df_long = df_standardised_02[["datetime", "year_month", "year", "month", "investigation", "number"]]

# Define paths for csv output
path_for_csv_wide = data_directories["interim_data_dir"] + "/df_wide" + ".csv"
path_for_csv_wide_timestamped = data_directories["interim_data_dir"] + "/df_wide" + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
path_for_csv_long = data_directories["interim_data_dir"] + "/df_long" + ".csv"
path_for_csv_long_timestamped = data_directories["interim_data_dir"] + "/df_long" + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"

# Export DataFrames to csv
df_wide.to_csv(path_for_csv_wide, index=False)
df_wide.to_csv(path_for_csv_wide_timestamped, index=False)

df_long.to_csv(path_for_csv_long, index=False)
df_long.to_csv(path_for_csv_long_timestamped, index=False)

printdf(df_wide)
printdf(df_long)


# Database directory for storing raw and processed data
sqlite_db_file = data_directories["database_dir"] + "/data_storage.sqlite"

# process_spreadsheet_into_sqlite(raw_data02, sheet_name=None, path_to_database=sqlite_db_file)

# Insert CSVs into SQLite database
real_path_to_wide_csv = realpath(expanduser(data_directories["interim_data_dir"] + "/df_wide.csv"))

real_path_to_long_csv = realpath(expanduser(data_directories["interim_data_dir"] + "/df_long.csv"))

process_csv_into_sqlite(real_path_to_wide_csv, path_to_database=sqlite_db_file)

process_csv_into_sqlite(real_path_to_long_csv, path_to_database=sqlite_db_file)

