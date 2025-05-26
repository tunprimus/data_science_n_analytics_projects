#!/usr/bin/env python3
from os.path import expanduser, realpath
from pathlib import Path

data_dir = Path("/path/to/some/logical/parent/dir")
data_dir = realpath(expanduser("/path/to/some/logical/parent/dir")
data_path = data_dir / "my_file.extension"

