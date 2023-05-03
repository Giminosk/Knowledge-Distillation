# Knowledge distillation

Implemetation of simple example of knowledge distillation. Knowledge distillation is a process in machine learning where a small, more efficient model is trained to mimic the behavior of a larger, more complex model. The goal is to transfer the knowledge learned by the larger model to the smaller one, while reducing its size and computational requirements. This technique is often used to improve the performance of resource-limited devices or to accelerate the inference time of models in production.

 - Reference: [paper](https://arxiv.org/abs/1503.02531)
 - Dataset: custom dataset for classification task
 - Teacher model: [Res-Net152](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html)
 - Student model: Conv-Net with tree convolution blocks and two fully-connected

## Teacher network pretrain results

<img src="/images/pretrain/accuracy.png" width="300" height="200"/> <img src="/images/pretrain/roc.png" width="300" height="200"/>

<img src="/images/pretrain/f1.png" width="300" height="200"/> <img src="/images/pretrain/loss.png" width="300" height="200"/>

## Student network train results
 - black line - with teacher
 - blue line - withous teacher

<img src="/images/train/accuracy.png" width="300" height="200"/> <img src="/images/train/roc.png" width="300" height="200"/>

<img src="/images/train/f1.png" width="300" height="200"/> <img src="/images/train/loss.png" width="300" height="200"/>
