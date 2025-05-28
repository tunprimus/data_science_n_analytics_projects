#!/usr/bin/env python3
import sqlite3
import sys
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
    pd.set_option("mode.copy_on_write", True)
from os.path import expanduser, realpath
from pathlib import Path
from ydata_profiling import ProfileReport

proj_dir_path = Path.cwd().parent.parent
proj_dir_name = Path.cwd().parent.parent.name
sys.path.append(realpath(expanduser(proj_dir_path)))

print(f"Project directory name is: {proj_dir_name}")

from src.get_data.data_directories import data_directories

# Get path to sqlite database
path_to_sqlite = data_directories["database_dir"] + "/data_storage.sqlite"

# SQL query
sql_select_all_query = """SELECT * FROM df_standardised"""

# Function to load DataFrame from Sqlite database
def load_from_sqlite_table(table_name, path_to_sqlite, sql_query):
    real_path_to_sqlite = realpath(expanduser(path_to_sqlite))

    with sqlite3.connect(real_path_to_sqlite) as conn:
        table_df = pd.read_sql(sql_query, conn)

    return table_df

ultrasound_turnover = load_from_sqlite_table("df_standardised", path_to_sqlite, sql_select_all_query)

print(ultrasound_turnover.info())

ultrasound_turnover.drop("added_on", axis=1, inplace=True)

ultrasound_turnover.dropna(inplace=True)

ultrasound_turnover.tail()

uss_turnover_profile_report = ProfileReport(ultrasound_turnover)

# path_to_report = str(proj_dir_path) + "/reports/uss_turnover_ydata_profile_report.html"
path_to_report = proj_dir_path / "/reports/uss_turnover_ydata_profile_report.html"
real_path_to_report = realpath(expanduser(path_to_report))

uss_turnover_profile_report.to_file(output_file=path_to_report)
