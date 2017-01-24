# CarND-Behavioral-Cloning-Mini

This repository contains basic python files of [ddrsmile/CarND-Behavioral-Cloning](https://github.com/ddrsmile/CarND-Behavioral-Cloning)

## Quick Start

### Generate Training Data
**use `--data_path` to set the folder which contain image data and driving log***
#### Model
```
time python data_gen.py --crop_from=53 --crop_to=133 --new_h=16 --new_w=64 --recovery_angle=0.25 --perturb=0.01 --flip=0.01 --gray_method='hsv'
```
#### Tiny Model
```
time python data_gen.py --suffix=tiny --new_h=16 --new_w=32 --recovery_angle=0.25 --perturb=0.01 --flip=0.01 --gray_method='hsv'
```

### Train Models
* `python model.py`
* `python model.tiny.py`

### Run Autonomous Driving
* `python drive.py model.json`
* `python drive.tiny.py model.tiny.json`

