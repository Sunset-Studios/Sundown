import { MLOps } from "../ops/ops.js";
import { Name } from "../names.js";

const mse_loss_name = 'mse_loss';

export class MSELoss {
  static initialize(layer) { }

  static forward(layer, input_tensor, target_tensor = null) {
    const props = layer.properties;
    layer.loss = MLOps.mse_loss(target_tensor, input_tensor, props.enabled_logging, Name.from(props.name || mse_loss_name));
    return input_tensor;
  }

  static backward(layer, grad_output_tensor, target_tensor = null) {
    return MLOps.mse_loss_prime(target_tensor, grad_output_tensor);
  }
} 