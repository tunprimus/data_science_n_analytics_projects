#!/usr/bin/env python3
from os.path import expanduser, realpath

# Define data directories
raw_data_dir = realpath(expanduser("../../data/00-raw"))
interim_data_dir = realpath(expanduser("../../data/01-interim"))
retraining_data_dir = realpath(expanduser("../../data/02-retraining"))
final_data_dir = realpath(expanduser("../../data/03-final"))
external_data_dir = realpath(expanduser("../../data/external"))
database_dir = realpath(expanduser("../../data/database"))

data_directories = {
    "raw_data_dir": raw_data_dir,
    "interim_data_dir": interim_data_dir,
    "retraining_data_dir": retraining_data_dir,
    "final_data_dir": final_data_dir,
    "external_data_dir": external_data_dir,
    "database_dir": database_dir,
}
