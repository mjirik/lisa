cd ..
call activate lisa
python lisa\update_stable.py
python lisa\organ_segmentation.py
call deactivate
