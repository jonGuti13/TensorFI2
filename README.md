### Introduction

This repository contains the code used to perform the fault injection campaign using the [TensorFI 2](https://github.com/DependableSystemsLab/TensorFI2) injector designed by [Dependable Systems Lab](https://github.com/DependableSystemsLab). The resault of the experiments can be found [here]().

The purpose of this repository is, therefore, to share, temporarily and until the [merge requests](https://github.com/DependableSystemsLab/TensorFI2/pulls) are not merged, the new features that have been added to the TensorFI 2 injector. The features are as follows:
- [Change]() the way in which YAML files are loaded.
- [Add]() the possibility of choosing the layer where you want to inject the faults if `Target` is `layer_states` and `Mode` is `single`.
- [Change]() the way to choose the index of the parameter where the fault is going to be injected to a more memmory-efficient manner.
- [Add]() the possibility, in models quantized with TensorFlow Lite, to inject faults in the weights (INT8) for the case in which `Target` is `layer_states` and `Mode` is `single`.
- [Add]() the possibility, in models quantized with TensorFlow Lite, to inject faults in the bias (INT32) for the case in which `Target` is `layer_states` and `Mode` is `single`.

For any questions that are not related to any of the cases described above, please post the [corresponding issue](https://github.com/DependableSystemsLab/TensorFI2/issues) in the original repository.

### Table of Contents

- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Dependencies](#dependencies)
- [Examples](#examples)
  - [sample.yaml for example 1: injecting a fault in an INT8 weight](#sampleyaml-for-example-1-injecting-a-fault-in-an-int8-weight)
  - [sample.yaml for example 2: injecting a fault in an INT32 bias](#sampleyaml-for-example-2-injecting-a-fault-in-an-int32-bias)
  - [Error correction](#error-correction)

### Dependencies

1. TensorFlow framework (tested with 2.12.0)

2. Python (tested with 3.8.6)

3. PyYaml (tested with v6.0)

4. Keras framework (tested with 2.12.0, part of TensorFlow)

5. numpy package (tested with 1.23.5, part of TensorFlow)

### Examples

Let's see two examples of how to inject a bit-flip into a layer state in a TensorFlow Lite quantized model.

Go to [experiments/layer-states](https://github.com/DependableSystemsLab/TensorFI2/blob/master/experiments/layer-states) and set the sample.yaml file in [experiments/layer-states/confFiles](https://github.com/DependableSystemsLab/TensorFI2/tree/master/experiments/layer-states/confFiles) with the following configuration:

#### sample.yaml for example 1: injecting a fault in an INT8 weight

    Target: layer_states
    Mode: single
    Type: bitflips
    Amount: 1
    Bit: N
    Layer: N
    Format: int8

#### sample.yaml for example 2: injecting a fault in an INT32 bias

    Target: layer_states
    Mode: single
    Type: bitflips
    Amount: 1
    Bit: N
    Layer: N
    Format: int32

For further understanding of what each label and values mean, navigate to [conf/](https://github.com/DependableSystemsLab/TensorFI2/tree/master/conf) and check out how to set the fault injection configuration for the tests or experiments you plan to run.

Run the test to observe the fault injection. At first, it will fail and we will explain why and how to correct it. For example, let's say we run the simple convolutional neural network example:

    python3 cnn-mnist-tflite.py ./tflite/cnn-trained-int8.tflite ./confFiles/sample.yaml ./ 1 10

`./` is the directory where you want to store the output of the run, `1` is the number of fault injections you want to run and `10` is the number of test inputs to evaluate each of the fault injection runs.

#### Error correction

As a consequence of how the quantized TensorFlow Lite models are opened, it is not straightforward to know which are the indexes of the trainable parameters of the model (the ones where the fault can be injected). This is better understood by inspecting the '/experiments/layer-states/tflite/cnn-trained-int8.tflite' example model as explained in 'experiments/layer-states/cnn-mnist-tflite.py'.

On the one hand, if I open /experiments/layer-states/tflite/cnn-trained-int8.tflite in [Netron](https://netron.app/), I see that the 'location' of the first Conv2D layer (32x5x5x1) is 1 and I guess that the location of the bias (32) of the first Conv2D layer is 2 because I see that the location of the second Conv2D layer (64x5x5x32) is 3. However, the location of the MaxPool2D is 4, so where is the bias (64) of the second Conv2D layer located?

On the other hand, if I open the model as done in 'src/tensorfi2.py' the sizes are:
- 40 for layer 3 --> The 10 32-bit bias of FullyConnected layer represented as 40 8-bit integers.
- 10240 for layer 4 --> The 10x1024 8-bit weights of FullyConnected layer.
- 256 for layer 5 --> The 64 32-bit bias of the second Conv2D layer represented as 256 8-bit integers.
- 51200 for layer 6 --> The 64x5x5x32 8-bit weights of the second Conv2D layer.
- 128 for layer 7 --> The 32 32-bit bias of the first Conv2D layer represented as 128 8-bit integers.
- 800 for layer 8 --> The 32x5x5x1 8-bit weights of the first Conv2D layer.

This is because, when applying full-integer quantization in TensorFlow Lite, bias are quantized to 32 bit integers, but as a consequence of how the model is opened, each bias is stored as 4 consecutive 8-bit elements in the whole bias 1D array, so the value of the first bias is calculated as follows: b[3] * 2^24 + b[2] * 2^16 + b[1] * 2^8 + b[0].

All in all, to solve the error, we have to [modify](https://github.com/jonGuti13/TensorFI2/blob/master/src/tensorfi2.py#L82) line 82 in layer_states by:

```
indexes = [4, 6, 8] #Extracted from above for INT8 or [3,5,7] for INT32 when the model is opened as done in 'src/tensorfi2.py'
random.shuffle(indexes)
layernum = indexes[0]
```