import { LayerType } from "./ml_types.js";
import { Tensor } from "./math/tensor.js";

/**
 * Shape rule for FullyConnected layers.
 *
 * A scalar output declaration means "replace the last axis with this many
 * units". A full-rank output shape is also valid if it preserves the input
 * leading axes.
 */
const fully_connected_shape_rule = Object.freeze({
  infer_output_shape(layer, input_shape) {
    const props = layer.properties ?? {};
    if (!Tensor.is_shape_valid(props.output_size)) {
      throw new Error(`FullyConnected layer ${layer.id} requires output dimensions.`);
    }

    const output_shape = Tensor.normalize_shape(props.output_size);

    configure_fully_connected(layer, input_shape, output_shape);

    return layer.properties.output_shape;
  },
});

/**
 * Default rule for layers that preserve tensor shape.
 *
 * This covers activation layers (`ReLU`, `Sigmoid`, `Softmax`, `Tanh`) and
 * most elementwise layers we may add later.
 */
const passthrough_shape_rule = Object.freeze({
  infer_output_shape(layer, input_shape) {
    if (!Tensor.is_shape_valid(input_shape)) {
      mark_shape_pending(layer);
      return null;
    }

    configure_passthrough_layer(layer, input_shape);
    return input_shape;
  },
});

/**
 * Central shape-rule registry.
 *
 * Add new shape-changing layers here. Layers not present in the registry use
 * the pass-through rule, which is correct for shape-preserving function layers
 * and keeps simple future elementwise layers working out of the box.
 */
const shape_rules = new Map([[LayerType.FULLY_CONNECTED, fully_connected_shape_rule]]);

/**
 * Determines the external source shape for this setup pass.
 *
 * The pass relies on a concrete queued sample because subnet
 * setup infers input shapes from actual data flowing into the root.
 * Peeks at the root input layer's queue without consuming data in order to infer shape of the input.
 *
 * @param {Object} root_layer
 * @returns {number[]|null}
 */
function get_external_source_shape(root_layer) {
  const first_sample = root_layer?.training_queue?.peek();
  if (root_layer?.training_queue?.length > 0 && first_sample?.input?.shape) {
    return Tensor.normalize_shape(first_sample.input.shape);
  }
  return null;
}

/**
 * Marks a layer as waiting for an input shape and clears derived shape state.
 *
 * Shape-dependent layers can be created, connected, moved, and inspected before
 * an input layer or queued sample exists. In that state, setup should defer
 * parameter initialization and avoid publishing stale shapes to downstream
 * layers until a real source shape is available.
 *
 * @param {Object} layer
 * @returns {void}
 */
function mark_shape_pending(layer) {
  layer.properties ??= {};

  layer.properties.shape_pending = true;
  layer.properties.inferred_input_shape = null;
  layer.properties.input_shape = null;
  layer.properties.output_shape = null;
  layer.properties.execution_input_shape = null;
  layer.properties.execution_output_shape = null;
}

/**
 * Configures a FullyConnected layer using appropriate shape semantics.
 *
 * FullyConnected transforms only the last axis. Leading axes are preserved as independent
 * slices, then represented internally as matrix rows:
 *
 *   [d0, d1, features] -> [d0 * d1, features]
 *   [d0, d1, units]    <- [d0 * d1, units]
 *
 * The same parameter matrix is shared for every leading-index slice and for
 * every batch item.
 *
 * @param {Object} layer
 * @param {number[]} input_shape
 * @param {number[]} output_shape
 * @returns {void}
 */
function configure_fully_connected(layer, input_shape, output_shape) {
  layer.properties ??= {};

  if (!Tensor.is_shape_valid(input_shape)) {
    mark_shape_pending(layer);

    const output_size = Tensor.last_dim(output_shape);
    layer.properties.output_size = output_size;
    layer.properties.output_shape = output_shape;

    return;
  }

  const input_leading_shape = Tensor.leading_shape(input_shape);
  let normalized_output_shape = output_shape;

  // `FullyConnected(256)` means "replace only the last axis with 256". A full-rank
  // declaration such as `[500, 500, 256]` is also allowed when the leading axes
  // match the input.
  if (output_shape.length === 1 && input_leading_shape.length > 0) {
    normalized_output_shape = [...input_leading_shape, output_shape[0]];
  }

  const output_leading_shape = Tensor.leading_shape(normalized_output_shape);

  if (!Tensor.shapes_equal(input_leading_shape, output_leading_shape)) {
    throw new Error(
      `FullyConnected output shape must preserve input leading axes. Input shape ${input_shape} cannot produce output shape ${output_shape}.`
    );
  }

  const input_size = Tensor.last_dim(input_shape);
  const output_size = Tensor.last_dim(normalized_output_shape);
  const row_count =
    input_leading_shape.length === 0 ? 1 : input_leading_shape.reduce((acc, dim) => acc * dim, 1);
  const execution_input_shape = [row_count, input_size];
  const execution_output_shape = [row_count, output_size];

  layer.properties.input_size = input_size;
  layer.properties.output_size = output_size;
  layer.properties.input_shape = input_shape;
  layer.properties.output_shape = normalized_output_shape;
  layer.properties.execution_input_shape = execution_input_shape;
  layer.properties.execution_output_shape = execution_output_shape;
  layer.properties.shape_pending = false;
}

/**
 * Configures a layer that preserves the runtime tensor shape.
 *
 * Activation layers preserve both their input and output feature dimensions.
 * Loss layers preserve the runtime tensor for backprop compatibility, but their
 * conceptual output dimension is one scalar loss value.
 *
 * @param {Object} layer
 * @param {number[]} input_shape
 * @param {number|null} output_size
 * @returns {void}
 */
function configure_passthrough_layer(layer, input_shape, output_size = null) {
  layer.properties ??= {};

  const input_size = Tensor.last_dim(input_shape);

  layer.properties.input_size = input_size;
  layer.properties.output_size = output_size ?? input_size;
  layer.properties.input_shape = input_shape;
  layer.properties.output_shape = input_shape;
  layer.properties.execution_input_shape = input_shape;
  layer.properties.execution_output_shape = input_shape;
  layer.properties.shape_pending = false;
}

/**
 * Returns the shape rule for a layer.
 *
 * @param {Object} layer
 * @returns {Object}
 */
function get_shape_rule(layer) {
  return shape_rules.get(layer.type) ?? passthrough_shape_rule;
}

/**
 * Infers a layer's input shape from all incoming source shapes.
 *
 * Root layers receive one synthetic external source shape. Non-root layers
 * receive their parent output shapes. The current graph does not have
 * concatenation/add/etc. shape rules yet, so multiple incoming shapes must
 * match.
 *
 * Stores the inferred input shape on a layer and invalidates setup-dependent
 * data when the shape changes.
 *
 * @param {Object} layer
 * @param {number[][]} shapes
 * @returns {number[]|null}
 */
function infer_input_shape(layer, shapes) {
  layer.properties ??= {};

  const normalized_shapes = shapes.map(Tensor.normalize_shape).filter((shape) => shape !== null);

  let input_shape = null;
  if (normalized_shapes.length !== 0) {
    const reference_shape = normalized_shapes[0];
    for (let i = 1; i < normalized_shapes.length; i++) {
      if (!Tensor.shapes_equal(reference_shape, normalized_shapes[i])) {
        throw new Error(
          `Cannot infer input shape for layer ${layer.id}: incoming shapes must match.`
        );
      }
    }
    input_shape = reference_shape;
  }

  const normalized_input_shape = Tensor.normalize_shape(input_shape);

  if (!Tensor.shapes_equal(layer.properties.inferred_input_shape, normalized_input_shape)) {
    layer.properties.inferred_input_shape = normalized_input_shape;
  }

  return normalized_input_shape;
}

/**
 * Reshapes a tensor only when a layer's execution shape differs from the
 * tensor's current shape.
 *
 * @param {Object|null} tensor
 * @param {number[]|null|undefined} shape
 * @returns {Object|null}
 */
export function reshape_tensor(tensor, shape) {
  if (!tensor || !Tensor.is_shape_valid(shape)) {
    return tensor;
  }

  const normalized_shape = Tensor.normalize_shape(shape);
  if (Tensor.shapes_equal(tensor.shape, normalized_shape)) {
    return tensor;
  }

  return tensor.reshape(normalized_shape);
}

/**
 * Performs a lazy subnet setup pass for shape-dependent layer state.
 *
 * The pass walks the subnet in topological order from the root. For each layer:
 *
 * 1. infer input shape from the root data or parent output shapes
 * 2. apply the layer's central shape rule
 * 3. configure shape-dependent layer internals
 * 4. publish the output shape for downstream children
 *
 * @param {number} root_id
 * @param {Object} graph
 * @returns {void}
 */
export function prepare_subnet_shapes(root_id, graph) {
  const subnet_layers = graph.get_subnet_and_shared_layers(root_id);

  const in_degree = new Map();
  const output_shapes = new Map();
  const external_source_shapes = new Map();

  // Count unresolved dependencies within this subnet. The graph already rejects
  // cycles, so this standard Kahn-style queue gives us parent-before-child order.
  for (let i = 0; i < subnet_layers.length; ++i) {
    const id = subnet_layers[i];
    const layer = graph.get_layer(id);
    in_degree.set(id, layer.parent_ids.length);

    if (layer.parent_ids.length === 0) {
      external_source_shapes.set(id, get_external_source_shape(layer));
    }
  }

  const queue = [];
  for (let i = 0; i < subnet_layers.length; i++) {
    const id = subnet_layers[i];
    if (in_degree.get(id) === 0) {
      queue.push(id);
    }
  }

  while (queue.length > 0) {
    const current_id = queue.shift();
    const current_layer = graph.get_layer(current_id);
    if (!current_layer) continue;

    const incoming_shapes = [];
    if (external_source_shapes.has(current_id)) {
      incoming_shapes.push(external_source_shapes.get(current_id));
    }

    for (let i = 0; i < current_layer.parent_ids.length; ++i) {
      const parent_id = current_layer.parent_ids.get(i);
      if (output_shapes.has(parent_id)) {
        incoming_shapes.push(output_shapes.get(parent_id));
      }
    }

    const inferred_input_shape = infer_input_shape(current_layer, incoming_shapes);

    const output_shape = get_shape_rule(current_layer).infer_output_shape(
      current_layer,
      inferred_input_shape
    );

    const published_output_shape = current_layer.properties.output_shape ?? output_shape;
    if (Tensor.is_shape_valid(published_output_shape)) {
      output_shapes.set(current_id, published_output_shape);
    }

    // Release children once every parent inside the subnet has published an
    // output shape.
    for (let i = 0; i < current_layer.child_ids.length; ++i) {
      const child_id = current_layer.child_ids.get(i);

      const new_degree = in_degree.get(child_id) - 1;
      in_degree.set(child_id, new_degree);

      if (new_degree === 0) {
        queue.push(child_id);
      }
    }
  }
}
