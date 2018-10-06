set PATH=%HOMEPATH%\Miniconda2;%HOMEPATH%\Miniconda2\Scripts;C:\Miniconda2\Scripts;C:\Miniconda2;%PATH%
set PATH=%HOMEPATH%\Miniconda3;%HOMEPATH%\Miniconda3\Scripts;C:\Miniconda3\Scripts;C:\Miniconda3;%PATH%


rem : get requirements for pip using wget and then remove this package
conda create -y -c mjirik -c SimpleITK -c menpo -c conda-forge -n lisa pywget wget
call activate lisa
python -m wget https://raw.githubusercontent.com/mjirik/lisa/master/requirements_pip.txt
conda uninstall -y pywget wget

rem :pygco is not compiled for osx now this is why it is not in meta.yaml
conda install -y -c mjirik -c SimpleITK -c menpo -c conda-forge --no-default-packages pip lisa pygco

rem :windows specific
conda install -y -c jmargeta scikit-fmm

pip install -r requirements_pip.txt
del requirements_pip.txt
rem :"w"mkdir %HOMEPATH%\projects

