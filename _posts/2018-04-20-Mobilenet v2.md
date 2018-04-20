---
layout: post
title:  MobileNetV2
subtitle:   Inverted Residuals and Linear Bottlenecks
date:   2018-04-20
author: gavin
header-img: img/shufflenet.jpg
catalog:    true
tags:
    - deep learning
---

> Author of paper: Mark Sandler Andrew Howard Menglong Zhu Andrey Zhmoginov Liang-Chieh Chen Google Inc.

# Abstract

In this paper we describe a new mobile architecture, MobileNetV2, that improves the state of the art performance of mobile models.

# Introduction

The drive to improve accuracy often comes at a cost: modern state of the art networks require high computational resources beyond the capabilities of many mobile and embedded applications.

This paper introduces a new neural network architecture that is specifically tailored for mobile and resource constrained environments, by significantly decreasing the number of operations and memory needed while retaining the same accuracy.

# Preliminaries, discussion and intuition

### Depthwise Separable Convolutions

<img src ='mobilenet1.png'>

### Linear Bottlenecks

MobileNetV1:

<img src='mobilenetv2_2.png' width='350'>

MobileNetV2:

<img src='mobilenetv2_3.png' width='350'>

<img src='mobilenetv2_1.png' width='450'>

### Inverted Residual Block

<img src='mobilenetv2_4.png' width='450'>

<img src='mobilenetv2_5.png' width='450'>

<img src='mobilenetv2_6.png' width='450'>

<img src='mobilenetv2_7.png' width='450'>

<img src='mobilenetv2_8.png' width='450'>





