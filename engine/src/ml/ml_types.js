export const ModelType = Object.freeze({
  NEURAL: 0,
});

export const OptimizerType = Object.freeze({
  ADAM: 0,
  SGD: 1,
});

export const LayerType = Object.freeze({
  FULLY_CONNECTED: 0,
  CONVOLUTIONAL: 1,
  POOLING: 2,
  DROPOUT: 3,
  MSE: 4,
  CROSS_ENTROPY: 5,
  BINARY_CROSS_ENTROPY: 6,
  RELU: 7,
  TANH: 8,
  SIGMOID: 9,
  SOFTMAX: 10,
});
