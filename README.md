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
```
df -h
lscpu
```

### X. Persistant disk - Adding additional storage place to the VM, and the price.
https://cloud.google.com/tpu/docs/storage-options#persistent-disk

https://cloud.google.com/tpu/docs/setup-persistent-disk

SSD provisioned space is $0.187 per GB a month. They cost 62 cents for every 100GB per day.

- 1. Add persistant disk to existing tpu-vm:
```
gcloud compute tpus tpu-vm attach-disk $tpu-name \
 --zone=$zone \
 --disk=$disk-name \
 --mode=$disk-mode

```
- 2. SSH into tpu-vm:
```
gcloud alpha compute tpus tpu-vm ssh tpu-name --zone zone
```
- 3. Format, mount, and set permission
```
#List the dists attached to the tpu-vm
sudo lsblk
#Format the disk
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
#Make a directory for the disk as a D drive
sudo mkdir -p D
#Mount the D drive
sudo mount -o discard,defaults /dev/sdb /mnt/disks/persist
#Set permissions for the persistent disk:
sudo chmod a+w D

```
-4. Detach disk
```
gcloud alpha compute tpus tpu-vm detach-disk $tpu-name --disk=$disk-name --zone=$zone
```
## Downloading data for pre-processing
- 1. Amazon AWS
Public amazon S3 bucket files can be downloaded using wget with http:
```
 wget http://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz
```
- 2. Extract tar gz
```
tar -xvzf train_mini.json.tar.gz
```



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
### X. How to run the TPU vm on background

command line - Keep processes running after SSH session disconnects - Unix & Linux Stack Exchange
Nohup command ensures that ssh runs after it disconnects.

### X. Now to run TPU vms like TPU clusters? (job submit, sending emails when finished, etc)
Sending emails: 
## Data
Data must be prepared in TFRecord format, and cloud TPU only takes gsbucket as the source. 

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

