#!/usr/bin/env python3
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import sys

try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
    pd.set_option("mode.copy_on_write", True)
from matplotlib import rcParams
from os.path import expanduser, realpath
from pathlib import Path

proj_dir_path = Path.cwd().parent.parent
proj_dir_name = Path.cwd().parent.parent.name
sys.path.append(realpath(expanduser(proj_dir_path)))

from src.config.config import CONSTANTS_DICT, global_directories
from src.utils.sqlite_mgmt import load_df_from_sqlite_table
from src.utils.statistical_analyses import calc_augmented_dickey_fuller, calc_kwiatkowski_phillips_schmidt_shin


# Set few figures parameters
rcParams["figure.figsize"] = CONSTANTS_DICT["FIG_SIZE"]
rcParams["figure.dpi"] = CONSTANTS_DICT["FIG_DPI"]
rcParams["savefig.format"] = CONSTANTS_DICT["SAVEFIG_FORMAT"]


# Retrieve DataFrame from Sqlite database
sql_select_all_query = """
SELECT *
FROM
    df_wide
;
"""

path_to_sqlite = Path(global_directories["data_dir"]).joinpath(
    "database", "data_storage.sqlite"
)

real_path_to_sqlite = realpath(expanduser(path_to_sqlite))

df_load = load_df_from_sqlite_table(
    "df_wide", real_path_to_sqlite, sql_select_all_query
)

# Drop redundant variables from the DataFrame
df = df_load.drop(["added_on"], axis=1)

# Drop rows with missing values
df.dropna(inplace=True)

df["datetime"] = pd.to_datetime(df["datetime"])

print(df.sample(CONSTANTS_DICT["RANDOM_SAMPLE_SIZE"]))

print(df.info())



## Time Series Analysis



## Anomaly Detection



## Regression Analysis

