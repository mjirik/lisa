conda create --no-default-packages -n lisa pip pywget numpy

activate lisa
python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_pip.txt
pip install -r requirements_pip.txt
rem :"w"mkdir %HOMEPATH%\projects