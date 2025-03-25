import { Tensor } from "../math/tensor.js";
import { MLOp, MLHop, MLOpParams, MLOpType, MLHopType } from "./op_types.js";
import { FrameAllocator, RandomAccessAllocator } from "../memory/allocator.js";
import { HopAPIAdapter } from "./hop_api_adapter.js";

const default_reset_op = { type: MLHopType.RESET_MODEL };

export class MLOpStore {
  observers = [];
  hop_handlers = new Map(); 
  ops = null;
  hops = null;
  params = null;
  transient = false;

  constructor(transient = false) {
    this.transient = transient;

    if (this.transient) {
      this.ops = new FrameAllocator(2048, MLOp);
      this.hops = new FrameAllocator(2048, MLHop);
    } else {
      this.ops = new RandomAccessAllocator(2048, MLOp);
      this.hops = new RandomAccessAllocator(2048, MLHop);
    }

    this.params = new MLOpParams(2048 * 16);
    this.hops_params = new MLOpParams(2048 * 16);

    this.register_hop_handler(MLHopType.CREATE_MODEL, HopAPIAdapter.create_model);
    this.register_hop_handler(MLHopType.ADD_LAYER, HopAPIAdapter.add_layer);
    this.register_hop_handler(MLHopType.ADD_ACTIVATION, HopAPIAdapter.add_activation);
    this.register_hop_handler(MLHopType.ADD_LOSS, HopAPIAdapter.add_loss);
    this.register_hop_handler(MLHopType.RESET_MODEL, HopAPIAdapter.clear_model);
  }

  register_observer(observer) {
    this.observers.push(observer);
  }
  
  unregister_observer(observer) {
    const index = this.observers.indexOf(observer);
    if (index !== -1) {
      this.observers.splice(index, 1);
    }
  }
  
  notify_observers(op) {
    for (let i = 0; i < this.observers.length; i++) {
      this.observers[i].on_model_changed(this, op);
    }
  }

  register_hop_handler(hop_type, handler) {
    this.hop_handlers.set(hop_type, handler);
  }

  unregister_hop_handler(hop_type) {
    this.hop_handlers.delete(hop_type);
  }

  append(other) {
    if (other.ops.length > 0) {
      this.ops.append(other.ops);
    }
    if (other.params.length > 0) {
      this.params.append(other.params);
    }
  }

  // ===== Low-level operations (operations processed by MLOps system) =====

  create_zero_tensor(shape, batch_size) {
    const op = this.ops.allocate();
    const a = Tensor.zeros(shape, batch_size);
    op.type = MLOpType.CREATE_ZERO_TENSOR;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = a.id;
    this.params.add(shape, shape.length);
    this.params.add(batch_size, 1);

    this.notify_observers(op);

    return a;
  }

  init_random(a, scale = 1) {
    const op = this.ops.allocate();
    op.type = MLOpType.INIT_RANDOM;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = a.id;
    this.params.add(scale, 1);

    this.notify_observers(op);
  }

  init_he(a) {
    const op = this.ops.allocate();
    op.type = MLOpType.INIT_HE;
    op.param_start = this.params.length;
    op.param_count = 0;
    op.result = a.id;

    this.notify_observers(op);
  }

  init_glorot(a, output_size) {
    const op = this.ops.allocate();
    op.type = MLOpType.INIT_GLOROT;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = a.id;
    this.params.add(output_size, 1);

    this.notify_observers(op);
  }

  mat_mul(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([a.shape[0], b.shape[1]], a.batch_size);
    op.type = MLOpType.MAT_MUL;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);

    this.notify_observers(op);

    return result;
  }

  transpose(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape.toReversed()], a.batch_size);
    op.type = MLOpType.TRANSPOSE;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);

    this.notify_observers(op);

    return result;
  }

  fill(a, value, offset = 0, size = null) {
    const op = this.ops.allocate();
    const result = a;
    op.type = MLOpType.FILL;
    op.param_start = this.params.length;
    op.param_count = 3;
    op.result = result.id;
    this.params.add(value, 1);
    this.params.add(offset, 1);
    this.params.add(size ?? a.length, 1);

    this.notify_observers(op);

    return result;
  }

  extend(a, add_dims = [], fill_value = 0) {
    const op = this.ops.allocate();
    const new_shape = a.shape.map((dim, index) =>
      index < add_dims.length ? dim + add_dims[index] : dim
    );
    const result = Tensor.zeros(new_shape, a.batch_size);
    op.type = MLOpType.EXTEND;
    op.param_start = this.params.length;
    op.param_count = 2 + add_dims.length;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(add_dims, add_dims.length);
    this.params.add(fill_value, 1);

    this.notify_observers(op);

    return result;
  }

  reshape(a, shape) {
    const op = this.ops.allocate();
    const result = Tensor.zeros(shape, a.batch_size);
    op.type = MLOpType.RESHAPE;
    op.param_start = this.params.length;
    op.param_count = 1 + shape.length;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(shape, shape.length);

    this.notify_observers(op);
    
    return result;
  }

  copy(a, b, offset = 0, size = null) {
    const op = this.ops.allocate();
    const result = b;
    op.type = MLOpType.COPY;
    op.param_start = this.params.length;
    op.param_count = 3;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(offset, 1);
    this.params.add(size ?? a.length, 1);

    this.notify_observers(op);

    return result;
  }

  clone(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.CLONE;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);

    this.notify_observers(op);

    return result;
  }

  add(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.ADD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);

    this.notify_observers(op);

    return result;
  }

  sub(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SUB;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);

    this.notify_observers(op);

    return result;
  }

  sub_assign(a, b) {
    const op = this.ops.allocate();
    const result = a;
    op.type = MLOpType.SUB_ASSIGN;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(b.id, 1);

    this.notify_observers(op);

    return result;
  }

  div(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.DIV;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);

    this.notify_observers(op);

    return result;
  }

  dot(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([1], a.batch_size);
    op.type = MLOpType.DOT;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);

    this.notify_observers(op);

    return result;
  }

  scale(a, b) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SCALE;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b, 1);

    this.notify_observers(op);

    return result;
  }

  relu(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.RELU;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);

    this.notify_observers(op);

    return result;
  }

  relu_backward(a, grad) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.RELU_BACKWARD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(grad.id, 1);

    this.notify_observers(op);

    return result;
  }

  tanh(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.TANH;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);

    this.notify_observers(op);

    return result;
  }

  tanh_backward(a, grad) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.TANH_BACKWARD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(grad.id, 1);

    this.notify_observers(op);

    return result;
  }

  sigmoid(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SIGMOID;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);

    this.notify_observers(op);

    return result;
  }

  sigmoid_backward(a, grad) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SIGMOID_BACKWARD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(grad.id, 1);

    this.notify_observers(op);

    return result;
  }

  fused_mul_add(a, b, c) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.FUSED_MUL_ADD;
    op.param_start = this.params.length;
    op.param_count = 3;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(b.id, 1);
    this.params.add(c.id, 1);

    this.notify_observers(op);

    return result;
  }

  mse_loss(target, output, enabled_logging = false, name = null) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([1], target.batch_size);
    op.type = MLOpType.MSE_LOSS;
    op.param_start = this.params.length;
    op.param_count = 4;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);
    this.params.add(enabled_logging, 1);
    this.params.add(name, 1);

    this.notify_observers(op);

    return result;
  }

  mse_loss_prime(target, output) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...target.shape], target.batch_size);
    op.type = MLOpType.MSE_LOSS_PRIME;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);

    this.notify_observers(op);

    return result;
  }

  softmax(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SOFTMAX;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);

    this.notify_observers(op);

    return result;
  }

  softmax_backward(a, grad) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape], a.batch_size);
    op.type = MLOpType.SOFTMAX_BACKWARD;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(a.id, 1);
    this.params.add(grad.id, 1);

    this.notify_observers(op);

    return result;
  }

  cross_entropy_loss(target, output, enabled_logging = false, name = null) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([1], target.batch_size);
    op.type = MLOpType.CROSS_ENTROPY_LOSS;
    op.param_start = this.params.length;
    op.param_count = 4;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);
    this.params.add(enabled_logging, 1);
    this.params.add(name, 1);

    this.notify_observers(op);

    return result;
  }

  cross_entropy_loss_prime(target, output) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...target.shape], target.batch_size);
    op.type = MLOpType.CROSS_ENTROPY_LOSS_PRIME;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);

    this.notify_observers(op);

    return result;
  }

  binary_cross_entropy_loss(target, output, enabled_logging = false, name = null) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([1], target.batch_size);
    op.type = MLOpType.BINARY_CROSS_ENTROPY_LOSS;
    op.param_start = this.params.length;
    op.param_count = 4;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);
    this.params.add(enabled_logging, 1);
    this.params.add(name, 1);

    this.notify_observers(op);

    return result;
  }

  binary_cross_entropy_loss_prime(target, output) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...target.shape], target.batch_size);
    op.type = MLOpType.BINARY_CROSS_ENTROPY_LOSS_PRIME;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(target.id, 1);
    this.params.add(output.id, 1);

    this.notify_observers(op);

    return result;
  }

  clip_l2_norm(a, max_norm) {
    const op = this.ops.allocate();
    const result = a;
    op.type = MLOpType.CLIP_L2_NORM;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(max_norm, 1);

    this.notify_observers(op);

    return result;
  }

  batch_reduce_mean(a) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...a.shape]);
    op.type = MLOpType.BATCH_REDUCE_MEAN;
    op.param_start = this.params.length;
    op.param_count = 1;
    op.result = result.id;
    this.params.add(a.id, 1);

    this.notify_observers(op);

    return result;
  }

  adam_moment_update(
    variable,
    m_tensor,
    v_tensor,
    grad,
    beta1,
    beta2,
    t,
    epsilon,
    learning_rate
  ) {
    const op = this.ops.allocate();
    const result = variable;
    op.type = MLOpType.ADAM_MOMENT_UPDATE;
    op.param_start = this.params.length;
    op.param_count = 8;
    op.result = result.id;
    this.params.add(m_tensor.id, 1);
    this.params.add(v_tensor.id, 1);
    this.params.add(grad.id, 1);
    this.params.add(beta1, 1);
    this.params.add(beta2, 1);
    this.params.add(t, 1);
    this.params.add(epsilon, 1);
    this.params.add(learning_rate, 1);

    this.notify_observers(op);

    return result;
  }

  write_tensor(tensor, index, value) {
    const op = this.ops.allocate();
    const result = tensor;
    op.type = MLOpType.WRITE_TENSOR;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(index, 1);
    this.params.add(value, 1);

    this.notify_observers(op);

    return result;
  }

  read_tensor(tensor, index) {
    const op = this.ops.allocate();
    const result = Tensor.zeros([...tensor.shape], tensor.batch_size);
    op.type = MLOpType.READ_TENSOR;
    op.param_start = this.params.length;
    op.param_count = 2;
    op.result = result.id;
    this.params.add(tensor.id, 1);
    this.params.add(index, 1);

    this.notify_observers(op);

    return result;
  }

  // ===== High-level operations (for external systems, never processed by MLOps) =====

  create_model(type, learning_rate, loss_fn, optimizer) {
    const hop = this.hops.allocate();
    hop.type = MLHopType.CREATE_MODEL;
    hop.param_start = this.hops_params.length;
    hop.param_count = 1;
    
    let result = null;
    if (this.hop_handlers.has(MLHopType.CREATE_MODEL)) {
      result = this.hop_handlers.get(MLHopType.CREATE_MODEL)(type, learning_rate, loss_fn, optimizer);
    }

    hop.result = result;

    this.hops_params.add(type, 1);

    this.notify_observers(hop);

    return result;
  }

  add_layer(type, input_size, output_size, model, options = {}, params = null) {
    const hop = this.hops.allocate();
    hop.type = MLHopType.ADD_LAYER;
    hop.param_start = this.hops_params.length;
    hop.param_count = 1;

    let result = null;
    if (this.hop_handlers.has(MLHopType.ADD_LAYER)) {
      result = this.hop_handlers.get(MLHopType.ADD_LAYER)(type, input_size, output_size, model, options, params);
    }

    hop.result = result;

    this.hops_params.add(type, 1);

    this.notify_observers(hop);

    return result;
  }

  add_activation(type, model = null) {
    const hop = this.hops.allocate();
    hop.type = MLHopType.ADD_ACTIVATION;
    hop.param_start = this.hops_params.length;
    hop.param_count = 1;
    
    let result = null;
    if (this.hop_handlers.has(MLHopType.ADD_ACTIVATION)) {
      result = this.hop_handlers.get(MLHopType.ADD_ACTIVATION)(type, model);
    }

    hop.result = result;

    this.hops_params.add(type, 1);

    this.notify_observers(hop);

    return result;
  }

  add_loss(type, enabled_logging = false, name = null, model = null) {
    const hop = this.hops.allocate();
    hop.type = MLHopType.ADD_LOSS;
    hop.param_start = this.hops_params.length;
    hop.param_count = 1;
    
    let result = null;
    if (this.hop_handlers.has(MLHopType.ADD_LOSS)) {
      result = this.hop_handlers.get(MLHopType.ADD_LOSS)(type, enabled_logging, name, model);
    }

    hop.result = result;

    this.hops_params.add(type, 1);

    this.notify_observers(hop);

    return result;
  }

  reset_model(model) {
    const hop = this.hops.allocate();
    hop.type = MLHopType.RESET_MODEL;
    
    let result = null;
    if (this.hop_handlers.has(MLHopType.RESET_MODEL)) {
      result = this.hop_handlers.get(MLHopType.RESET_MODEL)(model);
    }

    hop.result = result;

    this.notify_observers(hop);

    return result;
  }

  reset() {
    this.ops.reset();
    this.params.reset();
    this.hops.reset();
    this.hops_params.reset();

    this.notify_observers(default_reset_op);
  }
  
  export_to_json() {
    if (this.transient) return;
    // TODO: Implement
  }

  export_to_onnx() {
    if (this.transient) return;
    // TODO: Implement
  }
  
  static import_from_json(json_string) {
    // TODO: Implement
    return new MLOpStore();
  }

  static import_from_onnx(onnx_model) {
    // TODO: Implement
    return new MLOpStore();
  }
}

// Base class for observers that want to be notified of model changes
export class model_observer {
  on_model_changed(store, op) {
    // To be implemented by concrete observers
  }
}