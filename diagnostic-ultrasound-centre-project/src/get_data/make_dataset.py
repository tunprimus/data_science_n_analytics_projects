#!/usr/bin/env python3

import numpy as np
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
    pd.set_option("mode.copy_on_write", True)
from os.path import expanduser, realpath
from data_directories import data_directories

# Define constants
GOLDEN_RATIO = 1.618
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72
RANDOM_SAMPLE_SIZE = 13
RANDOM_SEED = 42

# Import raw data
raw_data = data_directories["raw_data_dir"] + "/000-real-ultrasound-patronage-data.xlsx"

df_raw = pd.read_excel(raw_data)
