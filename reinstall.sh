#!/usr/bin/env bash

set -eou pipefail

pushd extensions; cython -3 --cplus *.pyx; popd
pip uninstall -y cylimiter && pip install .
pytest tests