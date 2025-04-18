export const ModelType = Object.freeze({
  NEURAL: 0,
});

export const OptimizerType = Object.freeze({
  ADAM: 0,
  SGD: 1,
});

export const LayerType = Object.freeze({
  INPUT: 0,
  FULLY_CONNECTED: 1,
  CONVOLUTIONAL: 2,
  POOLING: 3,
  DROPOUT: 4,
  MSE: 5,
  CROSS_ENTROPY: 6,
  BINARY_CROSS_ENTROPY: 7,
  RELU: 8,
  TANH: 9,
  SIGMOID: 10,
  SOFTMAX: 11,
});

export const InputType = Object.freeze({
  IMAGE: 0,
  NUMERIC: 1,
  TEXT: 2,
});

