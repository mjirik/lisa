#!/bin/bash

# Skript je zavisly na souboru:
# /data/medical/processed/christmas2013/liver_volumetry_3mm_alpha30.yaml
# Z nej je kopii vytvoren yaml pro vyhodnoceni experimentu
#
# Vysledkem je adresarova struktura, kde vznikne adresar se skupinou
# experimentu. V nem budou yaml soubory, csv vysledky vyhodnoceni a adresar s
# merenimi jednotlivych fazi experimentu.

echo "param1: experimetn name"
echo "param2: directory in datadir"
echo ""
echo "Exaple: "
echo "    experiment_volumetry_evaluation.sh vs9.0mm_alpha30 christmas2013"

echo "Parametry $@"
mkdir -p ~/data/medical/processed/$2/$1
echo " ~/data/medical/processed/christmas2013/liver_volumetry_3mm_alpha30.yaml ~/data/medical/processed/$2/liver_volumetry_$1.yaml"
cp ~/data/medical/processed/christmas2013/liver_volumetry_3mm_alpha30.yaml ~/data/medical/processed/$2/liver_volumetry_$1.yaml
sed -i s/3mm_alpha30/$1/g ~/data/medical/processed/$2/liver_volumetry_$1.yaml
sed -i s/christmas2013/$2/g ~/data/medical/processed/$2/liver_volumetry_$1.yaml


cp ~/lisa_data/*$1*.pklz ~/data/medical/processed/$2/$1

python src/volumetry_evaluation.py -d -i ~/data/medical/processed/$2/liver_volumetry_$1.yaml -o ~/data/medical/processed/$2/eval_$1.csv
