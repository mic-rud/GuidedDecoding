# GuidedDecoding
Accompaning repository for the 2022 ICRA paper "Lightweight  Monocular  Depth  Estimation  through  Guided  Decoding"

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

[NYU Depth V2](https://drive.google.com/file/d/1hXvznCAa26bNBPGZJH1DI2siVxmQlm0W/view?usp=sharing)
[KITTI](https://drive.google.com/file/d/1EZ8hBSwiudUnpYvgC1-Z6iHSyeWaPRfx/view?usp=sharing)

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


Installing PyTorch and torchvision, refer to this post: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-11-now-available/72048

Installing torch2trt: https://github.com/NVIDIA-AI-IOT/torch2trt

You might need to increase SWAP memory for the tensorRT conversion to 4GB: https://github.com/JetsonHacksNano/resizeSwapMemory

## Trained weights
| Dataset  | Resolution | Model-Version |
| ------------- | -------------- | ------------- |
| NYU Depth V2  | 240x320 (Half) | [GuideDepth](https://drive.google.com/file/d/16oC0YW2yRNO_Sn4on0KsumkrhHtydikI/view?usp=sharing) |
| NYU Depth V2  | 240x320 (Half) | [GuideDepth-S](https://drive.google.com/file/d/1ZA80WcgKJsOWaOeBuSn3oupzKHV4eonv/view?usp=sharing)|
| NYU Depth V2  | 480x640 (Full) | [GuideDepth](https://drive.google.com/file/d/1TNTUUve5LHEv6ERN6v9aX2eYw1-a-4bO/view?usp=sharing)|
| NYU Depth V2  | 480x640 (Full) | [GuideDepth-S](https://drive.google.com/file/d/1HhKSpshT4RZe-wG6nSB2zwC-ooBwuVo9/view?usp=sharing)|
| KITTI         | 192x640 (Half) |  [GuideDepth](https://drive.google.com/file/d/1dqatUdck6nHPL0BOI5Xk9nKb_Ei954Hq/view?usp=sharing)|
| KITTI         | 192x640 (Half) |  TODO|
| KITTI         | 384x1280 (Full) |  [GuideDepth](https://drive.google.com/file/d/1rj629jYCjdGwXkW73-Lr868FPORFF2gR/view?usp=sharing)|
| KITTI         | 384x1280 (Full) |  TODO |

## Usage
```console
python3 inference.py --eval --model MODEL_NAME --resolution RESOLUTION --dataset DATASET --weights_path PATH_TO_WEIGHTS --save_results PATH_TO_RESULTS --test_path PATH_TO_TEST_DATA
```
By selecting from the following options:
```console
[RESOLUTION: full, half]
[DATASET: nyu_reduced, kitti]
```
