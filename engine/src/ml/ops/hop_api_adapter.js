import { ModelType, LossType, ActivationType, LayerType } from "../ml_types.js";

import { NeuralModel } from "../models/neural_model.js";

import { FullyConnected } from "../layers/fully_connected.js";

import { ReLU } from "../layers/relu.js";
import { Sigmoid } from "../layers/sigmoid.js";
import { Tanh } from "../layers/tanh.js";

import { MSELoss } from "../layers/mse_loss.js";
import { BinaryCrossEntropyLoss } from "../layers/binary_cross_entropy_loss.js";
import { CrossEntropyLoss } from "../layers/cross_entropy_loss.js";

export class HopAPIAdapter {
  static create_model(type, learning_rate, loss_fn, optimizer) {
    let model = null;

    if (type === ModelType.NEURAL) {
      model = new NeuralModel("sine_approximator", {
        learning_rate: learning_rate,
        optimizer: optimizer,
        loss_fn: loss_fn,
      });
    }

    return model;
  }

  static add_layer(type, input_size, output_size, model = null, options = {}, params = null) {
    let layer = null;

    if (type === LayerType.FULLY_CONNECTED) {
      layer = new FullyConnected(input_size, output_size, options, params);
    }

    if (model) {
      model.add(layer);
    }

    return layer;
  }

  static add_activation(type, model = null) {
    let activation = null;

    if (type === ActivationType.RELU) {
      activation = new ReLU();
    } else if (type === ActivationType.SIGMOID) {
      activation = new Sigmoid();
    } else if (type === ActivationType.TANH) {
      activation = new Tanh();
    }

    if (model) {
      model.add(activation);
    }

    return activation;
  }

  static add_loss(type, enabled_logging = false, name = null, model = null) {
    let loss = null;

    if (type === LossType.MSE) {
      loss = new MSELoss(enabled_logging, name);
    } else if (type === LossType.BINARY_CROSS_ENTROPY) {
      loss = new BinaryCrossEntropyLoss(enabled_logging, name);
    } else if (type === LossType.CROSS_ENTROPY) {
      loss = new CrossEntropyLoss(enabled_logging, name);
    }
    
    if (model) {
      model.add(loss);
    }

    return loss;
  }

  static clear_model(model) {
    if (!model) return;
    model.clear();
    return model;
  }
}
