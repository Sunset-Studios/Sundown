import { ModelType, LossType, ActivationType, LayerType, OptimizerType } from "../ml_types.js";

import { NeuralModel } from "../models/neural_model.js";

import { FullyConnected } from "../layers/fully_connected.js";

import { ReLU } from "../layers/relu.js";
import { Sigmoid } from "../layers/sigmoid.js";
import { Tanh } from "../layers/tanh.js";

import { MSELoss } from "../layers/mse_loss.js";
import { BinaryCrossEntropyLoss } from "../layers/binary_cross_entropy_loss.js";
import { CrossEntropyLoss } from "../layers/cross_entropy_loss.js";

import { Adam } from "../optimizers/adam.js";

export class HopAPIAdapter {
  static create_model(type, learning_rate, loss_fn, optimizer_type = null) {
    let model = null;

    if (type === ModelType.NEURAL) {
      model = new NeuralModel("neural_network", {
        learning_rate: learning_rate,
        loss_fn: loss_fn,
      });
    }

    if (optimizer_type !== null) {
      HopAPIAdapter.add_optimizer(optimizer_type, model);
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

  static add_optimizer(type, model = null, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    let optimizer = null;

    if (type === OptimizerType.ADAM) {
      optimizer = new Adam(beta1, beta2, epsilon);
    }

    if (model) {
      model.set_optimizer(optimizer);
    }

    return optimizer;
  }

  static clear_model(model) {
    if (!model) return;
    model.clear();
    return model;
  }
}
