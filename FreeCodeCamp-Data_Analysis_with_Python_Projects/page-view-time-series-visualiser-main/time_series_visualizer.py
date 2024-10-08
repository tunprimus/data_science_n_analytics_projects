import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

pd.set_option("mode.copy_on_write", True)

RANDOM_SAMPLE_SIZE = 13
GOLDEN_RATIO = 1.618
FIGURE_WIDTH = 30
FIGURE_HEIGHT = FIGURE_WIDTH / GOLDEN_RATIO

# Import data (Make sure to parse dates. Consider setting index column to 'date'.)
df = pd.read_csv("fcc-forum-pageviews.csv", index_col="date", parse_dates=True)
df.head()
df.tail()
df.info()
df.describe()
df.sample(RANDOM_SAMPLE_SIZE)

# Clean data
nor_val_low_limit = df["value"].quantile(0.025)
nor_val_high_limit = df["value"].quantile(0.975)
df = df[(df["value"] >= nor_val_low_limit) & (df["value"] <= nor_val_high_limit)]
df.head()
df.tail()
df.info()
df.describe()
df.sample(RANDOM_SAMPLE_SIZE)


def draw_line_plot():
    # Draw line plot
    fig = plt.figure(figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
    plt.plot(df, color="orange")
    plt.title("Daily freeCodeCamp Forum Page Views 5/2016-12/2019")
    plt.xlabel("Date")
    plt.ylabel("Page Views")
    plt.show()

    # Save image and return fig (don't change this part)
    fig.savefig('line_plot.png')
    return fig

def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df_bar = df.copy()
    print(df_bar)

    # Draw bar plot
    df_bar["year"] = [d.year for d in df_bar.index]
    df_bar["month"] = [d.strftime('%B') for d in df_bar.index]
    df_bar = df_bar.groupby(["year", "month"])["value"].mean()
    print(df_bar)
    
    fig = df_bar.unstack().plot(kind="bar", legend=True, figsize=(FIGURE_WIDTH,FIGURE_HEIGHT)).figure
    plt.xlabel("Years")
    plt.ylabel("Average Page Views")
    plt.legend(title="Months")

    # Save image and return fig (don't change this part)
    fig.savefig('bar_plot.png')
    return fig

def draw_box_plot():
    # Prepare data for box plots (this part is done!)
    df_box = df.copy()
    df_box.reset_index(inplace=True)
    df_box['year'] = [d.year for d in df_box.date]
    df_box['month'] = [d.strftime('%b') for d in df_box.date]

    # Draw box plots (using Seaborn)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
    sns.boxplot(x=df_box["year"], y=df_box["value"], ax=ax1)
    sns.boxplot(x=df_box["month"], y=df_box["value"], ax=ax2, order=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    # Set titles
    ax1.set_title("Year-wise Box Plot (Trend)")
    ax2.set_title("Month-wise Box Plot (Seasonality)")

    # Set labels
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Page Views")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Page Views")

    # Save image and return fig (don't change this part)
    fig.savefig('box_plot.png')
    return fig
