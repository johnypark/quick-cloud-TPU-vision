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
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   46 bits physical, 48 bits virtual
CPU(s):                          96
On-line CPU(s) list:             0-95
Thread(s) per core:              2
Core(s) per socket:              24
Socket(s):                       2
NUMA node(s):                    2
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           85
Model name:                      Intel(R) Xeon(R) CPU @ 2.00GHz
Stepping:                        3
CPU MHz:                         2000.154
BogoMIPS:                        4000.30
Hypervisor vendor:               KVM
Virtualization type:             full
L1d cache:                       1.5 MiB
L1i cache:                       1.5 MiB
L2 cache:                        48 MiB
L3 cache:                        77 MiB
NUMA node0 CPU(s):               0-23,48-71
NUMA node1 CPU(s):               24-47,72-95
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Mitigation; PTE Inversion
Vulnerability Mds:               Mitigation; Clear CPU buffers; SMT Host state unknown
Vulnerability Meltdown:          Mitigation; PTI
Vulnerability Mmio stale data:   Vulnerable: Clear CPU buffers attempted, no microcode; SMT Host state unknown
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Retpolines, IBPB conditional, IBRS_FW, STIBP conditional, RSB filling
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Mitigation; Clear CPU buffers; SMT Host state unknown
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse s
                                 se2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid 
                                 tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave a
                                 vx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single pti ssbd ibrs ibpb stibp fs
                                 gsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx sma
                                 p clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat md_clear arch
                                 _capabilities
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

