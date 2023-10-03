# MLPerf Inference – Intel-Habana Labs – GPT-J Calibration Details

## Calibration stage

We pass a set of calibration samples through the neural network executed in single-precision floating-point to obtain a profile of every activation tensor of the network. 
The profile consists of maximum absolute values over all samples. 

## Model quantization

The input activations and weights of all linear operators (linear and matmul operators), except for the final lm_head layer, are quantized to FP8-143. 
The output of the linear operators, as well as the inputs and outputs of all other operators (e.g., softmax, layer-norm, gelu) are in BFloat16 precision. 
In the conversion to FP8-143, we employed four values of exponent bias, specifically [3, 7, 11, 15]. These biases represent ranges of [+/- 0.9375, +/- 15, +/- 240, +/- 3840], respectively.
The weights are pre-quantized to FP8-143 (except for the lm_head layer, which is in BFloat16).
The following outlines how the exponent bias is selected for each activation and weight.

### Activations exponent-bias


Each quantized activation's exponent bias is determined by selecting the one whose range, multiplied by a backoff factor of 0.5, encompasses the measured range during the calibration stage. It's worth noting that none of the activations exceed the range of +/- 0.5 * 3840.

### Weights exponent-bias

The weights exponent-bias is chosen as the one that minimizes the mean-squared-error introduced by the conversion to FP8-143.
