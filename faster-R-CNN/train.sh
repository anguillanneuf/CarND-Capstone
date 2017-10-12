#!/bin/sh

pushd .
cd models/research

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
PATH_TO_YOUR_PIPELINE_CONFIG="../../training/faster_rcnn_resnet101_coco_traffic_light.config"
PATH_TO_TRAIN_DIR="../../training/"

# From the tensorflow/models/research/ directory
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}

popd
