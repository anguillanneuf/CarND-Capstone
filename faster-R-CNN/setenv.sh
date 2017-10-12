#!/bin/sh

# comment the miniconda3
sed -i '/anaconda3/s/^/#/g' ~/.bashrc
# install miniconda 2, we need python 2.7
wget https://repo.continuum.io/miniconda/Miniconda2-4.3.27.1-Linux-x86_64.sh
chmod 751 Miniconda2-4.3.27.1-Linux-x86_64.sh
./Miniconda2-4.3.27.1-Linux-x86_64.sh
source ~/.bashrc
# install protobuf compiler
sudo apt-get update
sudo apt-get install protobuf-compiler
# create environment
conda env create -f environment.yml
source activate py2

# download pretrained model
pushd .
cd training
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
tar -vxzf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
popd

# download ObjectDetection API
git clone https://github.com/tensorflow/models.git

# prepare the model
pushd .
cd models/research
protoc object_detection/protos/*.proto --python_out=.
popd


