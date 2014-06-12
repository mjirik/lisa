#!/bin/sh

# First parameter gives config file
echo "Parametry $@"
echo "Pocet parametru $#"

add=""
add2=""

if [ $# -eq 1 ]; then
    echo "in iff"
    add="-cf"
    add2="$1"
fi

echo "add $add"

python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig001.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig002.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig003.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig004.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig005.mhd-exp010-seeds.pklz
echo "5"
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig006.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig007.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig008.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig009.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig010.mhd-exp010-seeds.pklz
echo "10"
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig011.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig012.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig013.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig014.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig015.mhd-exp010-seeds.pklz
echo "15"
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig016.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig017.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig018.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig019.mhd-exp010-seeds.pklz
python lisa.py -ni -dd ~/data/medical/processed/spring2014/exp010-seeds/org-liver-orig020.mhd-exp010-seeds.pklz
