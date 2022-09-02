#!/usr/bin/env bash

## Change root path
cd "$(dirname "$0")/../" || exit

## Active virtual environment or create it if not exists
if [ -d venv ]; then
    source venv/bin/activate
else
    python3 -m venv venv
    source venv/bin/activate
    python -m pip install -U pip wheel setuptools pytest
    python -m pip install -r requirements.txt
fi

# Run test
exec python -m pytest "$@"
