# Resnet-family Convolutional Neural Networks for Image Classification in Tensorflow

In this repository you will find implementation of Resnet and their variations for image classification.  The current TensorFlow version supported is 2.2. Users need to convert their models to TensorFlow2 if they are currently based on TensorFlow V1.x, or run in compatibility mode.  Users can refer to the [TensorFlow User Guide](https://docs.habana.ai/projects/SynapeAI-Gaudi/en/latest/Tensorflow_User_Guide/Tensorflow_User_Guide.html) to learn how to migrate existing models to run on Gaudi.

## Models

The following Resnet based models are provided:

|    **Model**     |              **Link**              |
| ---------------- | ---------------------------------- |
| ResNet50/101/152 | [README](./Resnets/README.md)      |
| ResNeXt101       | [README](./Resnets/README.md#example-commands)      |
| SSD-ResNet34     | [README](./SSD_ResNet34/README.md) |
| DenseNet         | [README](./densenet_keras/README.md) |
| Mask R-CNN       | [README](./maskrcnn_matterport_demo/README.md) |
