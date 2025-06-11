#!/usr/bin/env python3

def univariate_stats(df):
    """
    Generate descriptive statistics and visualisations for each feature in a DataFrame.

    This function computes and returns a DataFrame containing a variety of univariate
    statistics for each feature (column) in the input DataFrame `df`. It calculates
    metrics such as the data type, count of non-missing values, number of missing values,
    number of unique values, and mode for all features. For numerical features, it
    additionally calculates minimum, lower boundary of normal (2.5 percentile), first quartile,
    median, third quartile, upper boundary of normal (97.5 percentile), maximum, mean,
    standard deviation, skewness, and kurtosis. It also creates a histogram for
    numerical features and a count plot for categorical features.

    Parameters
    ----------
    df (pd.DataFrame): The DataFrame for which univariate statistics are to be computed.

    Returns
    -------
    pd.DataFrame: A DataFrame where each row corresponds to a feature from the input
      DataFrame and columns contain the calculated statistics.
    
    Example
    -------
    univariate_stats(df)
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    # Avoid below due to bug in Fireducks
    # https://github.com/fireducks-dev/fireducks/issues/45
    # try:
    #     import fireducks.pandas as pd
    # except ImportError:
    #     import pandas as pd

    output_df = pd.DataFrame(
        columns=[
            "feature",
            "type",
            "count",
            "missing",
            "unique",
            "mode",
            "min",
            "lbn_2_5pct",
            "q1",
            "median",
            "q3",
            "ubn_97_5pct",
            "max",
            "mean",
            "std",
            "skew",
            "kurt",
        ]
    )
    # output_df.set_index("feature", inplace=True) # This only works in Pandas and not FireDucks
    for col in df.columns:
        # Calculate metrics that apply to all columns dtypes
        dtype = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        mode = df[col].mode()[0]
        if pd.api.types.is_numeric_dtype(df[col]):
            # Calculate metrics that apply only to numerical features
            min_ = df[col].min()
            lbn_2_5pct = df[col].quantile(0.025)
            q1 = df[col].quantile(0.25)
            median = df[col].median()
            q3 = df[col].quantile(0.75)
            ubn_97_5pct = df[col].quantile(0.975)
            max_ = df[col].max()
            mean = df[col].mean()
            std = df[col].std()
            skew = df[col].skew()
            kurt = df[col].kurt()
            output_df.loc[col] = [
                col,
                dtype,
                count,
                missing,
                unique,
                mode,
                min_,
                lbn_2_5pct,
                q1,
                median,
                q3,
                ubn_97_5pct,
                max_,
                mean,
                std,
                skew,
                kurt,
            ]
            sns.histplot(data=df, x=col)
        else:
            output_df.loc[col] = [
                col,
                dtype,
                count,
                missing,
                unique,
                mode,
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
            ]
            sns.countplot(data=df, x=col)
        plt.show()
    output_df.set_index("feature", inplace=True)
    try:
        return output_df.sort_values(by=["missing", "unique", "skew"], ascending=False)
    except Exception:
        return output_df

