import { Tensor } from "../math/tensor.js";
import { Optimizer } from "../optimizer.js";

export class Adam extends Optimizer {
  constructor(beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
    super();
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.t = 0; // Time step for bias correction.
    // Maps to store first moment (m) and second moment (v) for each variable (by its tensor id)
    this.m = new Map();
    this.v = new Map();
  }

  // Ensure the variable is registered so that its m and v tensors are allocated.
  #register_variable(variable) {
    if (!this.m.has(variable.id)) {
      // Create m and v tensors with the same shape (and batch size) as the variable.
      const m = Tensor.zeros(variable.shape, variable.batch_size);
      const v = Tensor.zeros(variable.shape, variable.batch_size);
      m.persistent = true;
      v.persistent = true;
      this.m.set(variable.id, m);
      this.v.set(variable.id, v);
    }
  }

  // Applies an update to a variable given its gradient.
  // This method uses in-place updates on the variable tensor.
  apply_gradients(variable, grad, learning_rate = 0.001, weight_decay = 0) {
    // Register variable if needed.
    this.#register_variable(variable);

    const m_tensor = this.m.get(variable.id);
    const v_tensor = this.v.get(variable.id);

    this.t += 1; // Increase timestep.

    // First perform the Adam update with just the gradient
    variable.adam_moment_update(
      m_tensor,
      v_tensor,
      grad,
      this.beta1,
      this.beta2,
      this.t,
      this.epsilon,
      learning_rate
    );

    // Then apply weight decay separately after the Adam update
    if (weight_decay > 0) {
      const decay = variable.scale(weight_decay * learning_rate);
      variable.sub_assign(decay);  // In-place subtraction
    }
  }
}
