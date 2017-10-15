### Set-up  

In order to reproduce the results in this repo, TensorFlow Object Detection API is needed as well as the pre-trained SSD Mobile Net on the MSCOCO dataset. Please refer to `setenv.sh` for more detail and guidance.  

The correct folder structure in addition to what's provided by the API is as follows:  

```
+ models  
    + research  
        + object_detection  
            + data
                - train_extra.record
                - test.record
                - tl_label_map.pbtxt
            - ssd_mobilenet_v1_tl.config
            + sd_mobilenet_v1_coco_11_06_2017
                - frozen_inference_graph.pb  
                - model.ckpt.data-00000-of-00001  
                - model.ckpt.meta  
                - graph.pbtxt  
                - model.ckpt.index  
            + train_dir  
            + eval_dir  
            + output_dir  
            - create_train_test_pickle.ipynb  
            - create_train_test_record.ipynb  
            - evaluate_model.ipynb  
```

### Preprocessing  
Train and test record are created from partial manual annotation of traffic light color on the API output of site images as well as complete manual cropping and annotation on sim and internet images.  

|Source|Color|Number|Cropping|Annotation|
|----|----|----|----|----|
|site|red|67|API|manual|
|site|green|67|API|manual|
|sim|red|20|manual|manual|
|sim|yellow|18|manual|manual|
|sim|green|21|manual|manual|
|internet|red|5|manual|manual|
|internet|yellow|3|manual|manual|
|internet|green|3|manual|manual|  

### Training  
Training was done on an AWS p2.xlarge EC2 machine. We used a batch size of 16 and trained for 5000 steps.  

### Evaluation
Tensorboard was used to monitor training and evaluation.  
