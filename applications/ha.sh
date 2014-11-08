#!/bin/bash
SCRIPT_PATH="${BASH_SOURCE[0]}";
cd `dirname ${SCRIPT_PATH}` > /dev/null
cd ..

python lisa/update_stable.py
python lisa/histology_analyser.py $@
