#!/bin/bash

# Load necessary modules
ml devel math python/3.12.1 py-scipy/1.16.0_py312 py-pytest/8.3.4_py312 py-pandas/2.2.3_py312 py-mne/1.8.1_py312 py-yasa/0.6.5_py312 > /dev/null 2>&1 || true

# Get the python from the modules
PYTHON=$(/usr/bin/which python3.12 2>/dev/null || which python)

# Add the src directory to the Python path
cd /home/users/chaisaen/infraslow
export PYTHONPATH="/home/users/chaisaen/infraslow/src:$PYTHONPATH"

# Run pytest
$PYTHON -m pytest "$@"
