# GuidedDecoding
Accompaning repository for the 2022 ICRA paper "Lightweight  Monocular  Depth  Estimation  through  Guided  Decoding"
The following sections will describe the usage:
# Training
## Preparing NYU Depth V2
We used a Subset of NYU Depth V2 designed and prepared by Alhashim et al. (https://github.com/ialhashim/DenseDepth)

To train, download the dataset linked in their repository and store it at an appropriate direction.

## Preparing KITTI (Training)
TODO : Download data set

## Training procedure
```console
run main.py --train --dataset DATASET --resolution RESOLUTION --model MODEL_NAME --data_path PATH_TO_TRAINING_DATA --num_workers=NUM_WORKERS --save_checkpoint PATH_TO_CHECKPOINTS
```

## Evaluation procedure (on GPU)
For the evaluation, download the already prepared testsets from here:

TODO
TODO

and unpack them to a desired location.

```console
run main.py --eval --dataset DATASET --resolution RESOLUTION --model MODEL_NAME --test_path PATH_TO_TEST_DATA --num_workers=NUM_WORKERS --save_results PATH_TO_RESULTS
```

Training and evaluation can be performed at once combining both arguments.

You can select from the following options:
```console
[RESOLUTION: full, half]
[DATASET: nyu_reduced, kitti]
```

# Inference and deployment
We performed our evaluation on the NVIDIA Jetson Nano and the NVIDIA Xavier NX, using the following dependencies:

Jetpack: 4.5.1
CUDA: 10.2
CUDNN: 8.0.0
Python: 3.6.9
tensorRT: 7.1.3

PyTorch: 1.8.0
torchvision: 0.9.1

torch2trt: 0.2.0

Installing PyTorch and torchvision, refer to this post:
Installing torch2trt: 

## Trained weights
TODO

## Usage
```console
python3 inference.py --eval --model MODEL_NAME --resolution RESOLUTION --dataset DATASET --weights_path PATH_TO_WEIGHTS --save_results PATH_TO_RESULTS --test_path PATH_TO_TEST_DATA
```
By selecting from the following options:
```console
[RESOLUTION: full, half]
[DATASET: nyu_reduced, kitti]
```
