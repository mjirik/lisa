# Instructions for contributors


## First steps (for University of West Bohemia students)

1. Create a GitHub account

2. Send your GitHub login to miroslav.jirik@gmail.com

3. Clone Lisa repository

        git clone git@github.com:mjirik/lisa.git
        
    or use [Github Desktop](https://desktop.github.com/)
    
4. Install anaconda and other dependencies with standard [installation process](https://github.com/mjirik/lisa/blob/master/INSTALL.md)

## Use `virtualenv`

* On Windows activate virtual env in `cmd`

        call activate lisa
    
* On Linux the default virtual env is used, so there is no other action required
    
## Git workflow

    git pull
    git commit
    git push


## Testing

Download test data with:

    lisa --get_sample_data
    
Install additionall packages

    conda install -c mjirik -c SimpleITK lisa nose coverage
    
Run tests in project directory

    nosetests --with-coverage --cover-html --cover-package=lisa

## Code climate

Test code beauty

    pep8

## Documentation

in `docs/conf.py` have to be listed all extern modules to generate documentation.
You can check [build log](https://readthedocs.org/projects/liver-surgery-analyser)
