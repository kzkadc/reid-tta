#!/bin/bash

PACKAGE_DIR=$(pwd)

cd
cp -r $PACKAGE_DIR/openunreid .
cp $PACKAGE_DIR/README.md .
cp $PACKAGE_DIR/requirements.txt .
cp $PACKAGE_DIR/setup.py .

pip install .

cd $PACKAGE_DIR/tools
CONFIG=$CONFIG PYTHON=python3 bash dist_train.sh $METHOD $WORK_DIR $ARGS
