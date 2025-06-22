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
from src.utils.plot_outliers import plot_outliers_by_quantile, plot_outliers_by_univariate_chauvenet
from src.utils.sqlite_mgmt import load_df_from_sqlite_table
from src.utils.univariate_stats import univariate_stats


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

# Drop redundant variables from the DataFrame
df = df_load.drop(["datetime", "added_on"], axis=1)

# Drop rows with missing values
df.dropna(inplace=True)

print(df.sample(CONSTANTS_DICT["RANDOM_SAMPLE_SIZE"]))

print(df.info())



#-----------------------------------------------------------------------#
# Segment & Subset Data into Groups Based on Different Features         #
#-----------------------------------------------------------------------#


df.groupby(["investigation"])["number", "year_month"].plot()
df_grouper = df.groupby(["investigation"])["year_month", "month", "number"].reset_index()

df_grouper.value_counts()

df_grouper.sum()

df_grouper.size()

df_grouper.describe()["number"]

df_grouper.describe()["number"]

df_grouper.describe()["number"]["mean"]

df_grouper.describe()["number"]["mean"].plot()

df_grouper.get_group("Abdominal")["number"].plot()


pivot_table_y_m_n = df.pivot_table(index="year", columns="month", values="number")
sns.heatmap(pivot_table_y_m_n, cmap='coolwarm', annot=True, fmt=".0f")

pivot_table_y_i_n = df.pivot_table(index="year", columns="investigation", values="number")
sns.heatmap(pivot_table_y_i_n, cmap='coolwarm', annot=True, fmt=".0f")

pivot_table_m_i_n = df.pivot_table(index="month", columns="investigation", values="number")
sns.heatmap(pivot_table_m_i_n, cmap='coolwarm', annot=True, fmt=".0f")





#-----------------------------------------------------------------------#
# Compare Results of Distribution & Dependency Analyses of These Groups #
#-----------------------------------------------------------------------#

sns.violinplot(data=pivot_table_m_i_n)
plt.title("Frequency of Investigations")
plt.ylabel("Number")
plt.xlabel("Type of Investigation")
plt.xticks(rotation=90)
plt.show()


sns.swarmplot(data=pivot_table_m_i_n)
plt.title("Frequency of Investigations")
plt.ylabel("Number")
plt.xlabel("Type of Investigation")
plt.xticks(rotation=90)
plt.show()


sns.violinplot(data=pivot_table_m_i_n, fill=False, linewidth=0.7)
sns.swarmplot(data=pivot_table_m_i_n)
plt.title("Frequency of Investigations")
plt.ylabel("Number")
plt.xlabel("Type of Investigation")
plt.xticks(rotation=90)
plt.show()
