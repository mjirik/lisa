#!/bin/bash

if [ "$1" = "patch" ]; then
    echo "pull, patch, push, push --tags"
    git pull
    bumpversion patch
    git push
    git push --tags
elif [ "$1" = "minor" ]; then
    echo "pull, patch, push, push --tags"
    git pull
    bumpversion minor
    git push
    git push --tags
elif [ "$1" = "major" ]; then
    echo "pull, patch, push, push --tags"
    git pull
    bumpversion major 
    git push
    git push --tags
elif [ "$1" = "stable" ]; then
    # if [ "$#" -ne 2 ]; then 
        # git tag
        # echo "Wrong number of arguments. Use two arguments like:"
        # echo "distr.sh stable v1.7"
    # else
        # git checkout master
        # current_version=`bumpversion --dry-run --list minor | grep new_version | sed -r s,"^.*=",,`
        # git pull
        # git tag -a "v$current_version" -m "new stable Lisa version"
        git push --tags
        git checkout stable
        git pull origin master
        git push
        git checkout master
        exit 1
    # fi
fi
# upload to pypi
python setup.py register sdist upload

# build conda and upload

rm -rf win-*
rm -rf linux-*
rm -rf osx-*

conda build -c mjirik -c SimpleITK .
conda convert -p all `conda build --output .`

binstar upload */*.tar.bz2

rm -rf win-*
rm -rf linux-*
rm -rf osx-*

