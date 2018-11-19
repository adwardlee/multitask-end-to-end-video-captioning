# multitask-end-to-end-video-captioning
with reinforcement learning

## End-to-End Video Captioning with Multitask Reinforcement Learning

*****

This repo holds the codes and models for the end-to-end captioning method presented on WACV 2019
**Temporal Action Detection with Structured Segment Networks**
Lijun Li, Boqing Gong

[[Arxiv Preprint]](http://arxiv.org/abs/1803.07950)
## Prerequisites
* This code requires [tensorflow1.1.0](https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl). The evaluation code is in Python, and you need to install [coco-caption evaluation](https://github.com/tylin/coco-caption) if you want to evaluate the model.

use the following to clone to your local machine
``` bash
git clone https://github.com/adwardlee/video2text.git
```
### Download Datasets

We support experimenting with two publicly available datasets for 
video captioning: MSVD & MSR-VTT.

* [MSVD](https://www.microsoft.com/en-us/download/confirmation.aspx?id=52422)
* [MSR-VTT](http://ms-multimedia-challenge.com/2016/dataset)

## Preprocess data
###  Extract all frames from videos
It needs to extract the frames by using `cpu_extract.py`. Then use `read_certrain_number_frame.py` to uniformly sample 5 frames from all frames of a video. At last use the `tf_feature_extract.py` to extract the inception-resnet-v2 features of frame.

## Training from scratch
use the `*_s2vt.py`. Before that, it needs to change the model path of evaluation function and some global parameters in the file. For example,
```
python tf_s2vt.py --gpu 0 --task train
```


## Testing existing models
### Evaluate models
use the `*_s2vt.py`. Before that, it needs to change the model path of evaluation function and some global parameters in the file. For example,
```
python tf_s2vt.py --gpu 0 --task evaluate
```

The MSVD models can be downloaded from [here](https://drive.google.com/open?id=199se09ycy1nMF7tCs9R1J-lIA1sHKcHi)
The MSR-VTT models can be downloaded from [here](https://drive.google.com/open?id=16relLI2XWjgoM2kPXN55u2IT23CrEyLz)



we also apply temporal attention in tensorflow
