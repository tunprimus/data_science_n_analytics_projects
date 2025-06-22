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
from os.path import expanduser, realpath
from pathlib import Path
from ydata_profiling import ProfileReport

proj_dir_path = Path.cwd().parent.parent
proj_dir_name = Path.cwd().parent.parent.name
sys.path.append(realpath(expanduser(proj_dir_path)))

from src.config.config import global_directories
from src.utils.sqlite_mgmt import load_df_from_sqlite_table
from src.utils.bivariate_stats import scatterplot, bar_chart, crosstab, bivariate_stats
from src.utils.univariate_stats import univariate_stats

from src.config.config import CONSTANTS_DICT

# Retrieve DataFrame from Sqlite database
sql_select_all_query = """SELECT * FROM df_long"""

path_to_sqlite = Path(global_directories["data_dir"]).joinpath("database", "data_storage.sqlite")

real_path_to_sqlite = realpath(expanduser(path_to_sqlite))

df_load = load_df_from_sqlite_table("df_long", real_path_to_sqlite, sql_select_all_query)


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

date_cols = pd.Index(["year-month", "year", "month"])

print(date_cols)

# Visualise the variables
sns.swarmplot(x="year", y="number", data=df)
sns.swarmplot(x="month", y="number", data=df)
sns.boxenplot(x="investigation", y="number", data=df)

# Descriptive statistics for each variable
univariate_stats(df[num_cols])
univariate_stats(df)


# Generate profile report
uss_turnover_profile_report = ProfileReport(df)

# path_to_report = str(proj_dir_path) + "/reports/uss_turnover_ydata_profile_report.html"
path_to_report = proj_dir_path / "/reports/uss_turnover_ydata_profile_report.html"
real_path_to_report = realpath(expanduser(path_to_report))

uss_turnover_profile_report.to_file(output_file=path_to_report)

