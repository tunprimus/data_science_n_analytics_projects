import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import realpath as realpath

try:
    from .utils.EDA_utilities import column_summary
except ImportError:
    real_path_to_eda_utilities = realpath("../utils/EDA_utilities")
    from real_path_to_eda_utilities import column_summary

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

df = pd.read_excel
