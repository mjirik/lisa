#!/bin/bash
SCRIPT_PATH="${BASH_SOURCE[0]}";
cd `dirname ${SCRIPT_PATH}` > /dev/null
cd ..

python src/update_stable.py
python src/histology_analyser.py $@
