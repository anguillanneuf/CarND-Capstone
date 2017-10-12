#!/bin/sh
# install miniconda
wget https://repo.continuum.io/miniconda/Miniconda2-4.3.27.1-Linux-x86_64.sh
chmod 751 Miniconda2-4.3.27.1-Linux-x86_64.sh
./Miniconda2-4.3.27.1-Linux-x86_64.sh
source ~/.bashrc
sudo apt-get install protobuf-compiler
conda env create -f environment.yml
source activate py2

pushd .
cd training
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -vxzf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
popd

git clone https://github.com/tensorflow/models.git

pushd .
cd models/research
protoc object_detection/protos/*.proto --python_out=.
popd


