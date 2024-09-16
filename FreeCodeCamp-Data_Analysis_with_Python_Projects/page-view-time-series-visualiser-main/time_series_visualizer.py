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





    # Save image and return fig (don't change this part)
    fig.savefig('line_plot.png')
    return fig

def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df_bar = None

    # Draw bar plot





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





    # Save image and return fig (don't change this part)
    fig.savefig('box_plot.png')
    return fig
