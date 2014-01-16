#!/bin/bash

echo "param1: experimetn name"
echo "param2: directory in datadir"
echo ""
echo "Exaple: "
echo "    experiment_volumetry_evaluation.sh vs9.0mm_alpha30 christmas2013"

echo "Parametry $@"
cp ~/data/medical/processed/$2/liver_volumetry_3mm_alpha30.yaml ~/data/medical/processed/$2/liver_volumetry_$1.yaml
sed -i s/3mm_alpha30/$1/g ~/data/medical/processed/$2/liver_volumetry_$1.yaml

mkdir ~/data/medical/processed/$2/$1

cp ~/lisa_data/*$1*.pklz ~/data/medical/processed/$2/$1

python experiments/volumetry_evaluation.py -d -i ~/data/medical/processed/$2/liver_volumetry_$1.yaml -o ~/data/medical/processed/$2/volumetry_$1.csv
