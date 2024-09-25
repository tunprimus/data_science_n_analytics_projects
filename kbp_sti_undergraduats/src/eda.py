import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import realpath as realpath
from ydata_profiling import ProfileReport
from eda_utilities import (
    column_summary,
    column_summary_plus,
    dtype_to_json,
    download_csv_json,
    json_to_dtype,
    dataframe_preview,
    numerical_columns_identifier,
    rename_columns,
    explore_nulls_nans,
    selective_fill_nans,
    explore_correlation,
    display_pairwise_correlation,
    iv_woe,
    column_categoriser,
    model_data_partitioner,
    model_data_preprocessor_full_return,
    feature_importance_sorted,
    get_feature_importance,
    individual_t_test,
)


# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)


# --------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------


FIGURE_HEIGHT = 10
GOLDEN_RATIO = 1.618
FIGURE_WIDTH = FIGURE_HEIGHT * GOLDEN_RATIO
FIGURE_DPI = 72
TEST_SIZE = 0.19
RANDOM_STATE_SEED = 42

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

real_path_to_source_data = realpath(
    "../data/00_raw/Awareness_of_STI_and_Sexual_Behaviour.xlsx"
)

real_path_to_interim_data01 = realpath(
    "../data/01_interim/awareness_of_sti_and_sexual_behaviour_renamed_columns.xlsx"
)

df_raw = pd.read_excel(real_path_to_source_data)

df_01 = pd.read_excel(real_path_to_interim_data01)



# --------------------------------------------------------------
# Profile data
# --------------------------------------------------------------

column_summary(df_raw)
column_summary_plus(df_raw)
dataframe_preview(df_raw)
numerical_columns_identifier(df_raw)


profile_00 = ProfileReport(df_01, title="KBA STI Undergraduate")
profile_00.to_notebook_iframe()
profile_00.to_file("awareness_of_sti_and_sexual_behaviour_01.html")

profile_01 = ProfileReport(
    df_01, title="awareness_of_sti_and_sexual_behaviour", explorative=True
)
profile_01.to_file("awareness_of_sti_and_sexual_behaviour_02.html")
