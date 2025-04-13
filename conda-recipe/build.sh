#!/bin/sh

git clone https://github.com/benedekrozemberczki/karateclub.git
cd karateclub
$PYTHON -m pip install .

cd ..
rm -rf karateclub
$PYTHON setup.py install --single-version-externally-managed --record=$RECIPE_DIR/record.txt
