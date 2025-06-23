#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats

try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
    pd.set_option("mode.copy_on_write", True)

from statsmodels.stats.power import TTestIndPower
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


def adjust_bins(series, num_bin, min_bin_percent=0.05):
    """
    Adjust the number of bins for a Pandas series such that each bin is at least a certain percentage of the original array.
    Parameters
    ----------
    series: pd.Series
        Input Pandas series
    num_bin: int
        Initial number of bins.
    min_bin_percent: float, optional
        Minimum percentage of the original array for each bin. Defaults to 0.05 (5%).
    Returns
    -------
    int: Adjusted number of bins
    """
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    pd.set_option("mode.copy_on_write", True)

    # Calculate the minimum number of samples per bin
    min_samples_per_bin = int(len(series) * min_bin_percent)
    # Calculate the adjusted number of bins
    adjusted_bins = min(num_bin, int(np.ceil(len(series) / min_samples_per_bin)))

    return adjusted_bins if (adjusted_bins < num_bin) else num_bin


# ------------------------------------------------------#
# Normality Tests                                      #
# ------------------------------------------------------#

def test_normality_via_shapiro(arr, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform the Shapiro-Wilk normality test on an array of data.

    Parameters
    ----------
    arr : array_like
        The data to test for normality.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    shapiro_stat : float
        The test statistic.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The test is calculated using scipy.stats.shapiro.
    """
    import matplotlib.pyplot as plt
    from statsmodels.graphics.gofplots import qqplot

    optim_bins = adjust_bins(arr, 20)

    shapiro_stat, p_value = sp_stats.shapiro(arr)

    if messages:
        plt.hist(arr, bins=optim_bins, density=True)
        plt.show()

        qqplot(arr, line="s", alpha=alpha_val)
        plt.show()

        if p_value < alpha_val:
            print(
                f"Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Shapiro-Wilk: {shapiro_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Shapiro-Wilk: {shapiro_stat:.{num_dp}f}"
            )

    return shapiro_stat, p_value


def test_normality_via_dagostino(arr, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform a D'Agostino's K^2 test for normality of a dataset.

    Parameters
    ----------
    arr : array_like
        The dataset to test for normality.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    dagostino_stat : float
        The test statistic.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The test is calculated using scipy.stats.normaltest.
    """
    import scipy.stats as sp_stats

    dagostino_stat, p_value = sp_stats.normaltest(arr)

    if messages:
        if p_value < alpha_val:
            print(
                f"Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for D’Agostino’s K^2: {dagostino_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for D’Agostino’s K^2: {dagostino_stat:.{num_dp}f}"
            )

    return dagostino_stat, p_value


def test_normality_via_anderson_darling(arr, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform the Anderson-Darling test for normality on an array of data.

    Parameters
    ----------
    arr : array_like
        The data to test for normality.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    result : scipy.stats._distn_infrastructure.AndersonResult
        The result of the Anderson-Darling test, containing the test statistic,
        critical values, and significance levels.

    Notes
    -----
    The test is calculated using scipy.stats.anderson. It is used to test if a sample
    of data comes from a population with a specific distribution.
    """
    import scipy.stats as sp_stats

    result = sp_stats.anderson(arr)

    if messages:
        for i in range(len(result.critical_values)):
            sig_level, crit_val = (
                result.significance_level[i],
                result.critical_values[i],
            )
            if result.statistic < result.critical_values[i]:
                print(
                    f"Reject the null hypothesis. p-value = {sig_level:.{num_dp}f} for Anderson-Darling: {crit_val:.{num_dp}f}"
                )
            else:
                print(
                    f"Fail to reject the null hypothesis. p-value = {sig_level:.{num_dp}f} for Anderson-Darling: {crit_val:.{num_dp}f}"
                )

    return result


# ------------------------------------------------------#
# Sample Size Tests                                    #
# ------------------------------------------------------#

def calc_sample_size(
    min_detect_effect,
    std_dev_val,
    effect_size=0.25,
    alpha_val=0.05,
    power=0.80,
    messages=True,
):
    """
    Calculate the required sample size for a two-sample t-test to detect a specified effect size.

    Parameters
    ----------
    min_detect_effect : float
        The minimum detectable effect size as a difference in means.
    std_dev_val : float
        The standard deviation of the data.
    effect_size : float, optional
        The expected effect size as a fraction of the standard deviation. Default is 0.25.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    power : float, optional
        The desired statistical power of the test. Default is 0.80.
    messages : bool, optional
        If True, print the sample size to the console. Default is True.

    Returns
    -------
    sample_size : int
        The required sample size.

    Notes
    -----
    The calculation is based on the formula given in [1]_, which is an approximation of the sample size required for a two-sample t-test to detect a specified effect size.
    """
    from statsmodels.stats.power import TTestIndPower

    analyser = TTestIndPower()

    if min_detect_effect and std_dev_val:
        if not effect_size:
            effect_size = min_detect_effect / std_dev_val

    sample_size = analyser.solve_power(
        effect_size=effect_size,
        alpha=alpha_val,
        power=power,
        ratio=1,
        alternative="two-sided",
    )

    if messages:
        print(
            f"The required sample size is {sample_size} to get an effect size of {effect_size}."
        )

    return sample_size


def calc_min_sample_size_for_ab_test(
    conv_rate_1,
    desired_effect_size,
    alpha_val=0.05,
    power=0.80,
    messages=True,
):
    """
    Calculate the minimum sample size required for an A/B test to detect a certain effect size.

    Parameters
    ----------
    conv_rate_1 : float
        The conversion rate of the control group.
    desired_effect_size : float
        The desired effect size to be detected.
    alpha_val : float, optional
        The significance level. Default is 0.05.
    power : float, optional
        The desired power of the test. Default is 0.80.
    messages : bool, optional
        If True, print the result to the console. Default is True.

    Returns
    -------
    min_sample_size : int
        The required minimum sample size.
    """
    import math
    import numpy as np
    import scipy.stats as sp_stats

    # Find Z_beta
    z_beta = sp_stats.norm.ppf(power)

    # Find Z_alpha
    z_alpha = sp_stats.norm.ppf(1 - alpha_val / 2)

    # Estimate minimum sample size
    conv_rate_2 = conv_rate_1 + desired_effect_size
    avg_prop = (conv_rate_1 + conv_rate_2) / 2
    variance = avg_prop * (1 - avg_prop)
    min_sample_size = math.ceil(
        2 * variance * (z_beta + z_alpha) ** 2 / desired_effect_size**2
    )

    if messages:
        print(f"The required minimum sample size is {min_sample_size}.")

    return min_sample_size


# ------------------------------------------------------#
# Correlation Tests                                    #
# ------------------------------------------------------#

def calc_pearson_corr(arr_1, arr_2, alpha_val=0.05, num_dp=4, messages=True):
    """
    Calculate the Pearson correlation coefficient and its p-value between two arrays.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    pearson_corr : float
        The Pearson correlation coefficient.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The Pearson correlation coefficient is calculated using scipy.stats.pearsonr.
    """
    import scipy.stats as sp_stats

    pearson_corr, p_value = sp_stats.pearsonr(arr_1, arr_2)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably dependent. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Pearson correlation: {pearson_corr:.{num_dp}f}"
            )
        else:
            print(
                f"Probably independent. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Pearson correlation: {pearson_corr:.{num_dp}f}"
            )

    return pearson_corr, p_value


def calc_spearman_corr(arr_1, arr_2, alpha_val=0.05, num_dp=4, messages=True):
    """
    Calculate the Spearman rank correlation coefficient and its p-value between two arrays.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    spearman_corr : float
        The Spearman rank correlation coefficient.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The Spearman rank correlation coefficient is calculated using scipy.stats.spearmanr.
    """
    import scipy.stats as sp_stats

    spearman_corr, p_value = sp_stats.spearmanr(arr_1, arr_2)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably dependent. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Spearman correlation: {spearman_corr:.{num_dp}f}"
            )
        else:
            print(
                f"Probably independent. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Spearman correlation: {spearman_corr:.{num_dp}f}"
            )

    return spearman_corr, p_value


def calc_kendall_corr(arr_1, arr_2, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform the Kendall rank correlation coefficient test on two arrays of data.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    kendall_corr : float
        The Kendall rank correlation coefficient.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The Kendall rank correlation coefficient is calculated using scipy.stats.kendalltau.
    """
    import scipy.stats as sp_stats

    kendall_corr, p_value = sp_stats.kendalltau(arr_1, arr_2)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably dependent. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Kendall-tau correlation: {kendall_corr:.{num_dp}f}"
            )
        else:
            print(
                f"Probably independent. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Kendall-tau correlation: {kendall_corr:.{num_dp}f}"
            )

    return kendall_corr, p_value


def calc_chi_squared(df, col_1, col_2, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform a Chi-Squared test for independence between two categorical variables in a DataFrame.

    This function calculates the Chi-Squared statistic and the p-value for a 2x2 contingency table
    created from two specified columns in a DataFrame. The test assesses whether the two variables
    are independent of each other.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    col_1 : str
        The name of the first column to be used in the contingency table.
    col_2 : str
        The name of the second column to be used in the contingency table.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the contingency table and the result of the hypothesis test to the console.
        Default is True.

    Returns
    -------
    chi_squared_stat : float
        The computed Chi-Squared statistic.
    p_value : float
        The p-value of the test.

    Raises
    ------
    ValueError
        If either of the specified columns is not found in the DataFrame or if either column
        does not represent a binary (2-level) categorical variable.
    """
    import scipy.stats as sp_stats
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
        pd.set_option("mode.copy_on_write", True)

    if col_1 not in df.columns or col_2 not in df.columns:
        raise ValueError(f"Columns {col_1} and/or {col_2} not found in the DataFrame.")

    if (df[col_1].nunique() != 2) or (df[col_2].nunique() != 2):
        raise ValueError("Chi-square test is only valid for 2x2 contingency tables.")
        return None

    contingency_table = pd.crosstab(
        df[col_1], df[col_2], margins=True, margins_name="Total"
    )

    chi_squared_stat, p_value, dof, expected = sp_stats.chi2_contingency(
        contingency_table
    )

    if messages:
        print(contingency_table)
        if p_value < alpha_val:
            print(
                f"Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for χ²: {chi_squared_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for χ²: {chi_squared_stat:.{num_dp}f}"
            )

    return chi_squared_stat, p_value


# ------------------------------------------------------#
# Stationary Tests                                     #
# ------------------------------------------------------#

def calc_augmented_dickey_fuller(arr, alpha_val=0.05, num_dp=4, messages=True):
    """
    Calculate the Augmented Dickey-Fuller (ADF) Unit Root Test on an array of data.

    Parameters
    ----------
    arr : array_like
        The data to test for stationarity.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    adf_stat : float
        The test statistic.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The ADF test is calculated using `statsmodels.tsa.stattools.adfuller`.
    """
    from statsmodels.tsa.stattools import adfuller

    adf_stat, p_value, used_lags, num_obs, icbest, resstore = adfuller(arr)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably stationary. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Augmented Dickey-Fuller Unit Root Test: {adf_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably non-stationary. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Augmented Dickey-Fuller Unit Root Test: {adf_stat:.{num_dp}f}"
            )

    return adf_stat, p_value


def calc_kwiatkowski_phillips_schmidt_shin(
    arr, alpha_val=0.05, num_dp=4, messages=True
):
    """
    Calculate the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Unit Root Test on an array of data.

    Parameters
    ----------
    arr : array_like
        The data to test for stationarity.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    kpss_stat : float
        The test statistic.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The KPSS test is calculated using `statsmodels.tsa.stattools.kpss`.
    """
    from statsmodels.tsa.stattools import kpss

    kpss_stat, p_value, lags, crit_values = kpss(arr)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably stationary. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Kwiatkowski-Phillips-Schmidt-Shin Unit Root Test: {kpss_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably non-stationary. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Kwiatkowski-Phillips-Schmidt-Shin Unit Root Test: {kpss_stat:.{num_dp}f}"
            )

    return adf_stat, p_value


# ------------------------------------------------------#
# Homogeneity of Variances Tests                       #
# ------------------------------------------------------#

def calc_barlett(*arr, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform Bartlett's test for homogeneity of variances.

    Parameters
    ----------
    *arr : array_like
        The arrays of data to be tested.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    bartlett_stat : float
        The test statistic for the Bartlett test.
    p_value : float
        The p-value of the test.

    Notes
    -----
    Bartlett's test is used to test the null hypothesis that all input samples are from populations with equal variances.
    The test is sensitive to departures from normality.
    """
    import scipy.stats as sp_stats

    bartlett_stat, p_value = sp_stats.bartlett(*arr)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably different distributions. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Bartlett test: {bartlett_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably the same distributions. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Bartlett test: {bartlett_stat:.{num_dp}f}"
            )

    return bartlett_stat, p_value


def calc_levene(*arr, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform Levene's test for homogeneity of variances.

    Parameters
    ----------
    *arr : array_like
        The arrays of data to be tested.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    levene_stat : float
        The test statistic for the Levene test.
    p_value : float
        The p-value of the test.

    Notes
    -----
    Levene's test is used to test the null hypothesis that all input samples are from populations with equal variances.
    It is more robust to departures from normality than Bartlett's test.
    """
    import scipy.stats as sp_stats

    levene_stat, p_value = sp_stats.levene(*arr)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably different distributions. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Levene test: {levene_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably the same distributions. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Levene test: {levene_stat:.{num_dp}f}"
            )

    return levene_stat, p_value


# ------------------------------------------------------#
# Parametric Statistical Hypothesis Tests              #
# ------------------------------------------------------#

def calc_student_t_test(
    arr_1, arr_2, alpha_val=0.05, equal_var=False, num_dp=4, messages=True
):
    """
    Calculate a two-sample t-test between two arrays.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    equal_var : bool, optional
        If True, the two arrays are assumed to have equal variances. Default is False.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    t_stat : float
        The t-statistic.
    p_value_2_tailed : float
        The p-value of the two-tailed test.

    Notes
    -----
    The t-test is calculated using scipy.stats.ttest_ind.
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    import scipy.stats as sp_stats

    t_stat, p_value_2_tailed = sp_stats.ttest_ind(arr_1, arr_2, equal_var=equal_var)

    if messages:
        if p_value_2_tailed < alpha_val:
            print(
                f"Reject the null hypothesis. p-value = {p_value_2_tailed:.{num_dp}f} for two-tailed T-statistic: {t_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Fail to reject the null hypothesis. p-value = {p_value_2_tailed:.{num_dp}f} for two-tailed T-statistic: {t_stat:.{num_dp}f}"
            )

    return t_stat, p_value_2_tailed


def calc_student_t_test_extended(
    arr_1,
    arr_2,
    alpha_val=0.05,
    equal_var=False,
    confidence_level=0.95,
    num_dp=4,
    messages=True,
):
    """
    Perform an extended two-sample t-test between two arrays and calculate the confidence interval for the mean difference.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    equal_var : bool, optional
        If True, the two arrays are assumed to have equal variances. Default is False.
    confidence_level : float, optional
        The confidence level for the interval. Default is 0.95.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    t_stat : float
        The t-statistic.
    p_value_2_tailed : float
        The p-value of the two-tailed test.
    mean_diff : float
        The difference in means between the two arrays.
    confidence_interval : list of float
        The lower and upper bounds of the confidence interval for the mean difference.

    Notes
    -----
    The t-test is calculated using scipy.stats.ttest_ind. The confidence interval is derived using the Welch-Satterthwaite equation for degrees of freedom.
    """
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    import scipy.stats as sp_stats

    # Calculate the t-test and p-value
    t_stat, p_value_2_tailed = sp_stats.ttest_ind(arr_1, arr_2, equal_var=equal_var)

    # Get the sizes of both samples
    arr_len_1 = len(arr_1)
    arr_len_2 = len(arr_2)

    # Get the means of both samples
    x_bar_1 = np.mean(arr_1)
    x_bar_2 = np.mean(arr_2)

    # Get the standard deviations for both groups
    std_dev_1 = np.std(arr_1, ddof=1)
    std_dev_2 = np.std(arr_2, ddof=1)

    # Difference in means
    mean_diff = x_bar_1 - x_bar_2

    # Standard error of the differences
    se_diff = np.sqrt((std_dev_1**2 / arr_len_1) + (std_dev_2**2 / arr_len_2))

    # Degree of freedom (Welch-Satterthwaite equation)
    dof_val = (se_diff**4) / (
        (
            (std_dev_1**2 / arr_len_1) ** 2 / (arr_len_1 - 1)
            + ((std_dev_2**2 / arr_len_2) ** 2 / (arr_len_2 - 1))
        )
    )

    # Get the 2-sided t-critical value for the desired confidence interval
    t_crit = sp_stats.t.ppf(1 - (1 - confidence_level) / 2, dof_val)

    # Calculate the upper and lower bounds for the above confidence interval
    ci_lower = mean_diff - (t_crit * se_diff)
    ci_upper = mean_diff + (t_crit * se_diff)

    if messages:
        if p_value_2_tailed < alpha_val:
            print(
                f"Reject the null hypothesis. p-value = {p_value_2_tailed:.{num_dp}f} for two-tailed T-statistic: {t_stat:.{num_dp}f}, ..."
            )
            print(
                f"... mean difference = {round(mean_diff, num_dp)}, {confidence_level}% confidence interval = {round(ci_lower, num_dp)}, {round(ci_upper, num_dp)}."
            )
        else:
            print(
                f"Fail to reject the null hypothesis. p-value = {p_value_2_tailed:.{num_dp}f} for two-tailed T-statistic: {t_stat:.{num_dp}f}"
            )

    return t_stat, p_value_2_tailed, mean_diff, [ci_lower, ci_upper]


def calc_paired_student_t_test(arr_1, arr_2, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform the paired Student's t-test to determine if two related arrays of data come from the same distribution.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    t_paired_stat : float
        The t-statistic for the paired Student's t-test.
    p_value_paired : float
        The p-value of the test.

    Notes
    -----
    The paired Student's t-test is calculated using scipy.stats.ttest_rel.
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    import scipy.stats as sp_stats

    t_paired_stat, p_value_paired = sp_stats.ttest_rel(
        arr_1, arr_2, equal_var=equal_var
    )

    if messages:
        if p_value_paired < alpha_val:
            print(
                f"Probably different distributions. Reject the null hypothesis. p-value = {p_value_paired:.{num_dp}f} for paired Student’s t-test: {t_paired_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably the same distributions. Fail to reject the null hypothesis. p-value = {p_value_paired:.{num_dp}f} for paired Student’s t-test: {t_paired_stat:.{num_dp}f}"
            )

    return t_paired_stat, p_value_paired


def calc_anova(*arr, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform the one-way ANOVA test to determine if k samples come from the same distribution.

    Parameters
    ----------
    *arr : array_like
        The arrays of data to be tested.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    f_stat : float
        The f-statistic for the ANOVA.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The one-way ANOVA test is calculated using scipy.stats.f_oneway.
    """
    import scipy.stats as sp_stats

    f_stat, p_value = sp_stats.f_oneway(*arr)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably different distributions. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for one-way ANOVA: {f_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably the same distributions. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for one-way ANOVA: {f_stat:.{num_dp}f}"
            )

    return f_stat, p_value


def calc_repeated_measures_anova(
    df,
    dep_var_str,
    subj_var_str,
    within_str_list,
    alpha_val=0.05,
    num_dp=4,
    messages=True,
):
    """
    Perform a repeated measures ANOVA on a DataFrame.

    This function calculates the repeated measures ANOVA using the specified
    dependent variable, subject variable, and within-subject factor(s). It
    assesses whether there are statistically significant differences between
    group means when the same subjects are used for each group.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    dep_var_str : str
        The name of the dependent variable column.
    subj_var_str : str
        The name of the subject identifier column.
    within_str_list : list of str
        A list containing the name(s) of the within-subject factor(s).
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    res : AnovaResults instance
        The fitted AnovaRM results object containing the test statistics and p-values.

    Notes
    -----
    The repeated measures ANOVA is calculated using statsmodels.stats.anova.AnovaRM.
    """
    from statsmodels.stats.anova import AnovaRM

    anovarm = AnovaRM(df, dep_var_str, subj_var_str, within=[within_str_list])
    res = anovarm.fit()

    if messages:
        if res.pvalues[0] < alpha_val:
            print(
                f"Probably different distributions. Reject the null hypothesis. p-value = {res.pvalues[0]:.{num_dp}f} for repeated measures ANOVA: {res.fvalue[0]:.{num_dp}f}"
            )
        else:
            print(
                f"Probably the same distributions. Fail to reject the null hypothesis. p-value = {res.pvalues[0]:.{num_dp}f} for repeated measures ANOVA: {res.fvalue[0]:.{num_dp}f}"
            )

    return res


# ------------------------------------------------------#
# Non-parametric Statistical Hypothesis Tests          #
# ------------------------------------------------------#

def calc_mann_whitney(arr_1, arr_2, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform the Mann-Whitney U test to determine if two arrays of data come from the same distribution.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    mannwhitneyu_stat : float
        The test statistic.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The Mann-Whitney U test is calculated using scipy.stats.mannwhitneyu.
    """
    import scipy.stats as sp_stats

    mannwhitneyu_stat, p_value = sp_stats.mannwhitneyu(arr_1, arr_2)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably different distributions. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Mann-Whitney U Test: {mannwhitneyu_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably the same distributions. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Mann-Whitney U Test: {mannwhitneyu_stat:.{num_dp}f}"
            )

    return mannwhitneyu_stat, p_value


def calc_wilcoxon(arr_1, arr_2, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform the Wilcoxon signed-rank test on two related arrays of data.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    wilcoxon_stat : float
        The test statistic for the Wilcoxon signed-rank test.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The Wilcoxon signed-rank test is a non-parametric statistical test that
    assesses whether the distributions of two related samples are different.
    It is calculated using scipy.stats.wilcoxon.
    """
    import scipy.stats as sp_stats

    wilcoxon_stat, p_value = sp_stats.wilcoxon(arr_1, arr_2)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably different distributions. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Wilcoxon Signed-Rank: {wilcoxon_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably the same distributions. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Wilcoxon Signed-Rank: {wilcoxon_stat:.{num_dp}f}"
            )

    return wilcoxon_stat, p_value


def calc_kruskal_wallis(arr_1, arr_2, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform the Kruskal-Wallis H-test on two independent arrays of data.

    Parameters
    ----------
    arr_1 : array_like
        The first array of data.
    arr_2 : array_like
        The second array of data.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    kruskal_stat : float
        The test statistic for the Kruskal-Wallis H-test.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The Kruskal-Wallis H-test is a non-parametric statistical test that
    assesses whether the distributions of two independent samples are
    different. It is calculated using scipy.stats.kruskal.
    """
    import scipy.stats as sp_stats

    kruskal_stat, p_value = sp_stats.kruskal(arr_1, arr_2)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably different distributions. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Kruskal-Wallis H: {kruskal_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably the same distributions. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Kruskal-Wallis H: {kruskal_stat:.{num_dp}f}"
            )

    return kruskal_stat, p_value


def calc_friedman(*arr, alpha_val=0.05, num_dp=4, messages=True):
    """
    Perform the Friedman test to determine if the distributions of multiple
    arrays are different.

    Parameters
    ----------
    *arr : array_like
        The arrays of data to test.
    alpha_val : float, optional
        The significance level for the test. Default is 0.05.
    num_dp : int, optional
        The number of decimal places to round the results to. Default is 4.
    messages : bool, optional
        If True, print the results to the console. Default is True.

    Returns
    -------
    friedman_stat : float
        The test statistic for the Friedman test.
    p_value : float
        The p-value of the test.

    Notes
    -----
    The Friedman test is calculated using scipy.stats.friedmanchisquare.
    """
    import scipy.stats as sp_stats

    friedman_stat, p_value = sp_stats.friedmanchisquare(*arr)

    if messages:
        if p_value < alpha_val:
            print(
                f"Probably different distributions. Reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Friedman test: {friedman_stat:.{num_dp}f}"
            )
        else:
            print(
                f"Probably the same distributions. Fail to reject the null hypothesis. p-value = {p_value:.{num_dp}f} for Friedman test: {friedman_stat:.{num_dp}f}"
            )

    return friedman_stat, p_value
