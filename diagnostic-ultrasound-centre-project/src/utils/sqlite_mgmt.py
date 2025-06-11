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

proj_dir_path = Path.cwd().parent.parent
proj_dir_name = Path.cwd().parent.parent.name
sys.path.append(realpath(expanduser(proj_dir_path)))

# Function to load DataFrame from Sqlite database
def load_df_from_sqlite_table(table_name, path_to_sqlite, sql_query):
    """
    Loads a pandas DataFrame from a specified SQLite database table.

    Parameters
    ----------
    table_name (str): The name of the table to load from.

    path_to_sqlite (str): The path to the SQLite database file.

    sql_query (str): The SQL query to execute to load the data.

    Returns
    -------
    pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    real_path_to_sqlite = realpath(expanduser(path_to_sqlite))

    with sqlite3.connect(real_path_to_sqlite) as conn:
        print(f"Loading DataFrame for `{proj_dir_name}` project")
        return pd.read_sql(sql_query, conn)
