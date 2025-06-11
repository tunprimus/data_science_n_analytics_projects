#!/usr/bin/env python3
import sys
from os.path import expanduser, realpath
from pathlib import Path

# Get the project directory/folder name and add it to system's path
proj_dir_path = Path.cwd().parent.parent
proj_dir_name = Path.cwd().parent.parent.name
sys.path.append(realpath(expanduser(proj_dir_path)))

# Define path to all top level folders
# business_problems_dir = realpath(expanduser(proj_dir_path / "/business_problems_R_n_D"))
business_problems_dir = Path(proj_dir_path).joinpath("business_problems_R_n_D").resolve()
data_dir = Path(proj_dir_path).joinpath("data").resolve()
models_dir = Path(proj_dir_path).joinpath("model").resolve()
notebooks_dir = Path(proj_dir_path).joinpath("notebooks").resolve()
references_dir = Path(proj_dir_path).joinpath("references").resolve()
reports_dir = Path(proj_dir_path).joinpath("reports").resolve()
src_dir = Path(proj_dir_path).joinpath("src").resolve()


# Define dictionary for all the top level directories and their paths
global_directories = {
    "business_problems_dir": business_problems_dir,
    "data_dir": data_dir,
    "models_dir": models_dir,
    "notebooks_dir": notebooks_dir,
    "references_dir": references_dir,
    "reports_dir": reports_dir,
    "src_dir": src_dir,
}


# Define useful constants
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO

# Define useful constants in a dictionary
CONSTANTS_DICT = {
    "GOLDEN_RATIO": 1.618033989,
    "FIG_WIDTH": 20,
    "FIG_HEIGHT": (FIG_WIDTH) / GOLDEN_RATIO,
    "FIG_SIZE": (FIG_WIDTH, FIG_HEIGHT),
    "FIG_DPI": 72,
    "RANDOM_SAMPLE_SIZE": 13,
    "RANDOM_SEED": 42,
    "FONT_SIZE": FIG_HEIGHT,
    "TITLE_SIZE": 23,
}
