import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Monkey patching numpy >= 1.24 in order to successfully import from other libraries
np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

RANDOM_SAMPLE_SIZE = 13

# 1
df = pd.read_csv("medical_examination.csv")
df.head()
df.info()
df.describe()
df_columns = df.columns
df.sample(RANDOM_SAMPLE_SIZE)

# 2
def mark_overweight(df):
    return np.where(((df["weight"] / (df["height"] / 100) ** 2) > 25), 1, 0)


df["overweight"] = df.apply(mark_overweight, axis=1)
df.sample(RANDOM_SAMPLE_SIZE)

# 3
def data_normaliser(df):
    return np.where((df <= 1), 0, 1)

target_df = df.loc[:, ["cholesterol", "gluc", "smoke", "alco", "active", "cardio"]]
buffer_df = df.loc[:, ["cholesterol", "gluc", "smoke", "alco", "active", "cardio"]].apply(data_normaliser)
buffer_df.sample(RANDOM_SAMPLE_SIZE)
print(buffer_df["cholesterol"].unique())

df.loc[:, ["cholesterol", "gluc", "smoke", "alco", "active", "cardio"]] = df.loc[:, ["cholesterol", "gluc", "smoke", "alco", "active", "cardio"]].apply(data_normaliser)
df.sample(RANDOM_SAMPLE_SIZE)


# 4
def draw_cat_plot():
    # 5
    df_cat = None

    # 6
    df_cat = None

    # 7

    # 8
    fig = None

    # 9
    fig.savefig("catplot.png")
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = None

    # 12
    corr = None

    # 13
    mask = None

    # 14
    fig, ax = None

    # 15

    # 16
    fig.savefig("heatmap.png")
    return fig
