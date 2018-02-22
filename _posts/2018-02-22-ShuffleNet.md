---
layout: post
title:  ShuffleNet
subtitle:   An Extremely Efficient Convolutional Neural Network for Mobile Devices
date:   2018-02-22
author: gavin
header-img: img/post-bg-coffee.jpeg
catalog:    true
tags:
    - deep learning
---

> Author of paper: Xiangyu Zhang, Xinyu Zhou, Mentxiao Lin, Jian Sun

# Abstract

ShuffleNet: designed for mobile devices with very limited computing power.

The new architecture utilizes two new operations:

- pointwise group convolution

- channel shuffle

### Group Convolution

GConv was first introduced in AlexNet for distributing the model over two GPUs, which can reduce FLOPs.

```
inputChannel = 256
outputChannel = 256
kernelSize = 3 * 3
numsofParams = 256 * 3 * 3 * 256

group = 8
# input channel of each group = 32
# output channel of each group = 32
numsofParams = 8 * 32 * 3 * 3 * 32
```
**Side Effect:** outputs from a certain channel are only derived from a small fraction of input channels.

outputs from a certain group only relate to the inputs within the group.

This property blocks information flow between channel groups and weakens representation.

![avatar](/img/ShuffleNet/GConv.png)

### Depthwise Separable Convolutions

```
inputChannel = 3
outputChannel = 256
numsofParams = 3 * 3 * 3 * 256 = 6912

# DW
numsofParams = 3 * 3 * 3 + 3 * 1 * 1 * 256 = 795
```
![avatar](/img/ShuffleNet/ShuffleNetUnit.png)





