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
    plot_multi_subplots,
    add_mean_median_mode_quartile_to_violin_plot,
    add_hatch_to_plot
)


# Monkey patching NumPy for compatibility with version >= 1.24
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)
plt.style.use(realpath("../../apa_enhanced.mplstyle"))

# --------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------


FIGURE_HEIGHT = 20
GOLDEN_RATIO = 1.618
FIGURE_WIDTH = FIGURE_HEIGHT * GOLDEN_RATIO
FIGURE_DPI = 300
TEST_SIZE = 0.19
RANDOM_STATE_SEED = 42

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

real_path_to_source_data_raw = realpath(
    "../data/00_raw/Awareness_of_STI_and_Sexual_Behaviour.xlsx"
)

real_path_to_interim_data01 = realpath(
    "../data/01_interim/awareness_of_sti_and_sexual_behaviour_renamed_columns.xlsx"
)

real_path_to_source_data = realpath(
    "../data/00_raw/awareness_of_sti_and_sexual_behaviours-response02.xlsx"
)


df_raw = pd.read_excel(real_path_to_source_data_raw)

df_01 = pd.read_excel(real_path_to_interim_data01)

df = pd.read_excel(real_path_to_source_data)



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


# --------------------------------------------------------------
# Quick data visualisation via violin plots
# --------------------------------------------------------------

df.head()
df.info()
df.describe()
column_summary(df)
column_summary_plus(df)
dataframe_preview(df)
numerical_columns_identifier(df)
explore_nulls_nans(df)
explore_correlation(df)
column_categoriser(df)

print(df["level"])
df["level_str"] = df["level"].dropna().astype(int).map(lambda x: str(x) + "L")
print(df["level_str"])

sns.violinplot(data=df, y=df["age_yr"])

add_hatch_to_plot(sns.violinplot(data=df, y=df["age_yr"]))

sns.violinplot(data=df, y=df["age_yr"], hue=df["sex"])
sns.violinplot(data=df, y=df["age_yr"], hue=df["sex"], split=True, gap=.025, inner_kws=dict(box_width=15, whis_width=2, color=".8"))

add_hatch_to_plot(sns.violinplot(data=df, y=df["age_yr"], hue=df["sex"], split=True, gap=.025, inner_kws=dict(box_width=15, whis_width=2, color=".8")))

sns.violinplot(data=df, x=df["level"], y=df["age_yr"], hue=df["sex"])
sns.violinplot(data=df, x=df["level"], y=df["age_yr"], hue=df["sex"], split=True, gap=.025, inner="quart")

add_hatch_to_plot(sns.violinplot(data=df, x=df["level"], y=df["age_yr"], hue=df["sex"], split=True, gap=.025, inner_kws=dict(box_width=5, whis_width=2, color="r")))

add_mean_median_mode_quartile_to_violin_plot(df, col_x="age_yr", hue="sex", split=True, gap=.025)
add_mean_median_mode_quartile_to_violin_plot(df, col_x="level_str", col_y="age_yr", hue="sex", split=True, gap=.025)

