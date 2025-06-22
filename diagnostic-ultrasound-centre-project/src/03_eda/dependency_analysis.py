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
from src.utils.bivariate_stats import bar_chart, bivariate_stats, crosstab, scatterplot
from src.utils.plot_outliers import (
    plot_outliers_by_quantile,
    plot_outliers_by_univariate_chauvenet,
)
from src.utils.sqlite_mgmt import load_df_from_sqlite_table

# Set few figures parameters
rcParams["figure.figsize"] = CONSTANTS_DICT["FIG_SIZE"]
rcParams["figure.dpi"] = CONSTANTS_DICT["FIG_DPI"]
rcParams["savefig.format"] = CONSTANTS_DICT["SAVEFIG_FORMAT"]

# Retrieve DataFrame from Sqlite database
sql_select_all_query = """SELECT * FROM df_long"""

path_to_sqlite = Path(global_directories["data_dir"]).joinpath(
    "database", "data_storage.sqlite"
)

real_path_to_sqlite = realpath(expanduser(path_to_sqlite))

df_load = load_df_from_sqlite_table(
    "df_long", real_path_to_sqlite, sql_select_all_query
)


df_load.sample(CONSTANTS_DICT["RANDOM_SAMPLE_SIZE"])

# Drop redundant variables from the DataFrame
df = df_load.drop(["datetime", "added_on"], axis=1)

# Drop rows with missing values
df.dropna(inplace=True)

print(df.sample(CONSTANTS_DICT["RANDOM_SAMPLE_SIZE"]))

print(df.info())

# Numerical variables identification
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

num_cols = num_cols.drop(["year", "month"])

print(num_cols)

# Datetime variables identification
date_cols = df.select_dtypes(include=["datetime64"]).columns

date_cols = pd.Index(["year_month", "year", "month"])

print(date_cols)

# Visualisation and comparison of each numerical variable if in wide format
# for i, first in enumerate(num_cols):
#     for j, second in enumerate(num_cols):
#         if first == second:
#             continue
#         scatterplot(df, first, second)

for feat in num_cols:
    bar_chart(df, feat, "month")

bivariate_stats(df, "month")
bivariate_stats(df, "year_month")
bivariate_stats(df, "year")
bivariate_stats(df, "investigation")

# Use bivariate_stats to compare all variables
for feat in df.columns:
    print(f"Starting analysis for {feat}")
    print(bivariate_stats(df, feat))
    print(f"End analysis for {feat}")
