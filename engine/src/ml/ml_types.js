export const ModelType = Object.freeze({
  NEURAL: 0,
});

export const LossType = Object.freeze({
  MSE: 0,
  CROSS_ENTROPY: 1,
  BINARY_CROSS_ENTROPY: 2,
});

export const LayerType = Object.freeze({
  FULLY_CONNECTED: 0,
  CONVOLUTIONAL: 1,
  POOLING: 2,
});

export const ActivationType = Object.freeze({
  RELU: 0,
  TANH: 1,
  SIGMOID: 2,
  SOFTMAX: 3,
});
