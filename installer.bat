set PATH=%PATH%;%HOMEPATH%\Miniconda2;%HOMEPATH%\Miniconda2\Scripts;C:\Miniconda2\Scripts;C:\Miniconda2
conda create -y -c mjirik -c SimpleITK --no-default-packages -n lisa pip pywget pip lisa

call activate lisa
python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_pip.txt
pip install -r requirements_pip.txt
del requirements_pip.txt
rem :"w"mkdir %HOMEPATH%\projects

rem :windows specific
conda install -c jmargeta scikit-fmm
