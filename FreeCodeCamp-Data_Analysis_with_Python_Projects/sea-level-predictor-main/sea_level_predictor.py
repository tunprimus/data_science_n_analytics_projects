import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

pd.set_option("mode.copy_on_write", True)

RANDOM_SAMPLE_SIZE = 13
GOLDEN_RATIO = 1.618
FIGURE_WIDTH = 30
FIGURE_HEIGHT = FIGURE_WIDTH / GOLDEN_RATIO

def draw_plot():
    # Read data from file
    df = pd.read_csv("epa-sea-level.csv")
    df.head()
    df.tail()
    df.info()
    df.describe()
    df.sample(RANDOM_SAMPLE_SIZE)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH,FIGURE_HEIGHT))
    ax.scatter(x="Year", y="CSIRO Adjusted Sea Level", data=df, color="blue", label="original data")
    plt.title("Rise in Sea Level")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Level (inches)")
    plt.show()


    # Create first line of best fit
    # Use the `linregress` function from `scipy.stats` to get the slope and y-intercept of the line of best fit. Plot the line of best fit over the top of the scatter plot. Make the line go through the year 2050 to predict the sea level rise in 2050.
    slope, intercept, r_value, p_value, stderr = linregress(df["Year"], df["CSIRO Adjusted Sea Level"])
    range_of_years = pd.Series(range(1880, 2050))
    ax.plot(range_of_years, intercept + slope*range_of_years, color="red", label="first line of best fit")
    fig

    # Create second line of best fit
    # Plot a new line of best fit just using the data from year 2000 through the most recent year in the dataset. Make the line also go through the year 2050 to predict the sea level rise in 2050 if the rate of rise continues as it has since the year 2000.
    df_2000 = df[df["Year"] >= 2000]
    slope, intercept, r_value, p_value, stderr = linregress(df_2000["Year"], df_2000["CSIRO Adjusted Sea Level"])
    range_of_years2 = pd.Series(range(2000, 2050))
    ax.plot(range_of_years2, intercept + slope*range_of_years2, color="green", label="second line of best fit")

    # Add labels and title
    ax.set(xlabel="Year", ylabel="Sea Level (inches)", title="Rise in Sea Level")
    fig

    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()