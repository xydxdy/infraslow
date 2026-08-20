#!/bin/bash
set -e

# Install dependencies in the compute environment
python -m pip install -q -e ".[dev]" 2>&1 | grep -v "already satisfied" || true

# Run the tests
python -m pytest "$@"
