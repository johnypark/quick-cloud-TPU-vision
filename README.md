# quick-cloud-TPU-vision
Quick cloud TPU solutions for computer vision. Credit to TPU Research Cloud Program

## Introduction 

This repo is a specifed niche for computer vision applications using the cloud TPU. 

I received access to cloud TPUs through TPU Research Cloud Program for computer vision research, and decided to open-source my scripts.

TRC program:
https://sites.research.google/trc/about/

### X. Setting up the storage for training - gsbucket and Kaggle

### X. Pricing estimation of gsbucket

### X. Creating a TPU VM and the specification of TPUs

### X. Adding additional storage place to the VM, and the price.

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

