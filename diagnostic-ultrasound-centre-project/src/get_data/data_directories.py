#!/usr/bin/env python3
from os.path import expanduser, realpath

# Define data directories
raw_data_dir = realpath(expanduser("../../data/00-raw"))
interim_data_dir = realpath(expanduser("../../data/01-interim"))
processed_data_dir = realpath(expanduser("../../data/02-processed"))
final_data_dir = realpath(expanduser("../../data/03-final"))
external_data_dir = realpath(expanduser("../../data/external"))

data_directories = {
    "raw_data_dir": raw_data_dir,
    "interim_data_dir": interim_data_dir,
    "processed_data_dir": processed_data_dir,
    "final_data_dir": final_data_dir,
    "external_data_dir": external_data_dir,
}
