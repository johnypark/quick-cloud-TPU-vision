# quick-cloud-TPU-vision
Quick cloud TPU solutions for computer vision. Credit to TPU Research Cloud Program

## Introduction 

This repo is a specifed niche for computer vision applications using the cloud TPU. 

I received access to cloud TPUs through TPU Research Cloud Program for computer vision research, and decided to open-source my scripts.

TRC program:
https://sites.research.google/trc/about/

## Getting Ready

### X. Setting up the storage space for training - gsbucket and Kaggle

- Kaggle updates their gsbucket address every Tuesday. Therefore you need to set up a checkpoint if the training gets longer than a week, if you are using Kaggle. 
- Not sure cloud TPU could utilze private Kaggle dataset. --> We should check on this.  
### X. The pricing estimation of gsbucket

### X. Creating a TPU VM and the specification of TPUs

### X. Persistant disk - Adding additional storage place to the VM, and the price.
https://cloud.google.com/tpu/docs/storage-options#persistent-disk
https://cloud.google.com/tpu/docs/setup-persistent-disk

### X. TPU detection
https://stackoverflow.com/questions/59289014/how-to-check-if-tpu-device-type-is-v2-or-v3
Above does not work for TPU cloud, only works for colab TPU. Has not found any way to display the TPU version in TPU cloud
```
import tensorflow as tf
import os

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver('local')  # TPU detection
  #print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)
```
## Models

### X. List of model backbones

### X. How backbones are imported 

### X. How to add layers after the penultimate layer

## Training

### X. Usage of checkpoints
- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
### X. Saving models
```
serialize model to json
json_model = model.to_json()
#save the model architecture to JSON file
with open('fashionmnist_model.json', 'w') as json_file:
    json_file.write(json_model)
```
    
- https://www.tensorflow.org/api_docs/python/tf/keras/models/model_from_json
### X. Saving model weights
model.save_weights('model.h5')

### X. Saving csv
- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger

