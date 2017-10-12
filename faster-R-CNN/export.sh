#!/bin/sh
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
PATH_TO_YOUR_PIPELINE_CONFIG="../../training/faster_rcnn_resnet101_coco_traffic_light.config"
PATH_TO_TRAIN_DIR="../../training/"
PATH_TO_EVAL_DIR="../../"


# From the tensorflow/models/research/ directory
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --checkpoint_dir=${PATH_TO_TRAIN_DIR} \
    --eval_dir=${PATH_TO_EVAL_DIR}

# From tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory output_inference_graph.pb
