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

from src.config.config import global_directories
from src.utils.sqlite_mgmt import load_df_from_sqlite_table

sql_select_all_query = """SELECT * FROM df_standardised"""

path_to_sqlite = Path(global_directories["data_dir"]).joinpath("database", "data_storage.sqlite")

real_path_to_sqlite = realpath(expanduser(path_to_sqlite))

df = load_df_from_sqlite_table("df_standardised", real_path_to_sqlite, sql_select_all_query)
