#!/usr/bin/env python3
import sys
from os.path import expanduser, realpath
from pathlib import Path

# Get the project directory/folder name and add it to system's path
proj_dir_path = Path.cwd().parent.parent
proj_dir_name = Path.cwd().parent.parent.name
sys.path.append(realpath(expanduser(proj_dir_path)))

# Define path to all top level folders
data_dir = realpath(expanduser(proj_dir_name + "/data"))
docs_dir = realpath(expanduser(proj_dir_name + "/docs"))
models_dir = realpath(expanduser(proj_dir_name + "/model"))
notebooks_dir = realpath(expanduser(proj_dir_name + "/notebooks"))
references_dir = realpath(expanduser(proj_dir_name + "/references"))
reports_dir = realpath(expanduser(proj_dir_name + "/reports"))
src_dir = realpath(expanduser(proj_dir_name + "/src"))

# Define dictionary for all the top level directories and their paths
top_directories = {
    "data_dir": data_dir,
    "docs_dir": docs_dir,
    "models_dir": models_dir,
    "notebooks_dir": notebooks_dir,
    "references_dir": references_dir,
    "reports_dir": reports_dir,
    "src_dir": src_dir,
}

# Define useful constants in a dictionary
CONSTANTS_DICT = {
    "GOLDEN_RATIO": 1.618033989,
    "FIG_WIDTH": 20,
    "FIG_HEIGHT": FIG_WIDTH / GOLDEN_RATIO,
    "FIG_SIZE": (FIG_WIDTH, FIG_HEIGHT),
    "FIG_DPI": 72,
    "RANDOM_SAMPLE_SIZE": 13,
    "RANDOM_SEED": 42,
    "FONT_SIZE": FIG_HEIGHT,
    "TITLE_SIZE": 23,
}
