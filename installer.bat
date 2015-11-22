conda create -y --no-default-packages -n lisa pip pywget numpy

call activate lisa
python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_pip.txt
pip install -r requirements_pip.txt
conda install -y -c SimpleITK -c mjirik lisa
rem :"w"mkdir %HOMEPATH%\projects
