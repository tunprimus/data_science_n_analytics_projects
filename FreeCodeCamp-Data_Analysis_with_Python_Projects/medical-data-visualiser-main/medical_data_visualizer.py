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
GOLDEN_RATIO = 1.618
FIGURE_WIDTH = 30
FIGURE_HEIGHT = FIGURE_WIDTH / GOLDEN_RATIO

# 1
# Import the data and the preliminary profiling
df = pd.read_csv("medical_examination.csv")
df.head()
df.tail()
df.info()
df.describe()
df_columns = df.columns
df.sample(RANDOM_SAMPLE_SIZE)
for col in df_columns[7:]:
    print(df[col].unique())

# 2
# Always convert hot-encoded values to integers to prevent headache with the groupby function later
# Function to mark value as overweight
def mark_overweight(df):
    return np.where(((df["weight"] / (df["height"] / 100) ** 2) > 25), 1, 0)

# Create column for overweight
df["overweight"] = df.apply(mark_overweight, axis=1).astype("int")
df.info()
for col2 in df_columns[7:]:
    print(df[col2].unique())
df.sample(RANDOM_SAMPLE_SIZE)

# 3
# Function to normalise values as int 0 or 1
def data_normaliser(col):
    # return np.where((val <= 1), 0, 1)
    df[col] = df[col].apply(lambda x: 1 if x > 1 else 0)

target_df = df.loc[:, ["cholesterol", "gluc", "smoke", "alco", "active", "cardio"]]
buffer_df = df.loc[:, ["cholesterol", "gluc", "smoke", "alco", "active", "cardio"]].apply(data_normaliser, axis=1).astype("int")
buffer_df.sample(RANDOM_SAMPLE_SIZE)
print(buffer_df["cholesterol"].unique())

# Apply above normalisation function to the DataFrame
data_normaliser("cholesterol")
data_normaliser("gluc")

# Check if normalisation was successful
df_columns = df.columns
for col3 in df_columns[7:]:
    print(df[col3].unique())
df.sample(RANDOM_SAMPLE_SIZE)
df.info()
df.describe()


# 4
# Create a chart of categorical values
def draw_cat_plot():
    # 5
    # Create a new DataFrame with pd.melt from certain target columns
    target_columns = ["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    df_cat = pd.melt(df, id_vars="cardio", value_vars=target_columns)

    # 6
    # Use groupby function to generate DataFrame for plot
    df_cat = df_cat.groupby(["cardio", "variable", "value"]).size().reset_index()
    df_cat.rename(columns={0: "total"}, inplace=True)

    # 7
    # Create the plot
    graph = sns.catplot(data=df_cat, kind="bar", x="variable", y="total", hue="value", col="cardio")

    # 8
    # Save the figure object from above chart
    fig = graph.fig

    # 9
    # Save the image from the plot object
    fig.savefig("catplot.png")
    return fig


# 10
# Draw the Heat Map in the `draw_heat_map` function
def draw_heat_map():
    # 11
    # Clean the data in the `df_heat` variable by filtering out the following patient segments that represent incorrect data
    filter_dp_normal = df["ap_lo"] <= df["ap_hi"]
    h_low_limit = df["height"].quantile(0.025)
    h_high_limit = df["height"].quantile(0.975)
    w_low_limit = df["weight"].quantile(0.025)
    w_high_limit = df["weight"].quantile(0.975)
    df_heat = df[filter_dp_normal & (df["height"] >= h_low_limit) & (df["height"] <= h_high_limit) & (df["weight"] >= w_low_limit) & (df["weight"] <= w_high_limit)]
    print(df_heat)

    # 12
    # Calculate the correlation matrix and store it in the `corr` variable
    corr = df_heat.corr()

    # 13
    # Generate a mask for the upper triangle and store it in the `mask` variable.
    # numpy.triu applies to the upper triangle of an array.
    # mask = np.triu(corr)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    # Set up the `matplotlib` figure.
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))

    # 15
    # Plot the correlation matrix using the method provided by the `seaborn` library import: `sns.heatmap()`.
    # seaborn.heatmap(data, *, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None, square=False, xticklabels='auto', yticklabels='auto', mask=None, ax=None, **kwargs)
    sns.heatmap(data=corr, mask = mask, fmt=".1f", linewidth=0.25, annot=True)

    # 16
    fig.savefig("heatmap.png")
    return fig
