import { Layer } from "./layer.js";
import { LayerType } from "./ml_types.js";
import { Tensor } from "./math/tensor.js";

function infer_fully_connected_output_shape(layer, input_shape) {
  const output_shape = Tensor.normalize_shape(layer.properties?.output_size);
  if (!output_shape) {
    return null;
  }

  const normalized_input_shape = Tensor.normalize_shape(input_shape);
  if (!normalized_input_shape) {
    return output_shape;
  }

  const input_leading_shape = Tensor.leading_shape(normalized_input_shape);
  if (output_shape.length === 1 && input_leading_shape.length > 0) {
    return [...input_leading_shape, output_shape[0]];
  }

  const output_leading_shape = Tensor.leading_shape(output_shape);
  return Tensor.shapes_equal(input_leading_shape, output_leading_shape) ? output_shape : null;
}

function infer_layer_output_shape(layer, input_shape) {
  if (layer.type === LayerType.FULLY_CONNECTED) {
    return infer_fully_connected_output_shape(layer, input_shape);
  }

  return Tensor.normalize_shape(input_shape);
}

function infer_subnet_output_shapes(root_id, input_shape) {
  const subnet_layers = Layer.get_subnet_and_shared_layers(root_id);
  const subnet_layer_ids = new Set(subnet_layers);
  const in_degree = new Map();
  const output_shapes = new Map();
  const external_source_shapes = new Map();
  const queue = [];

  for (let i = 0; i < subnet_layers.length; i++) {
    const id = subnet_layers[i];
    const layer = Layer.get(id);
    if (!layer) {
      continue;
    }

    in_degree.set(id, layer.parent_ids.length);

    if (layer.parent_ids.length === 0) {
      external_source_shapes.set(id, id === root_id ? Tensor.normalize_shape(input_shape) : null);
      queue.push(id);
    }
  }

  while (queue.length > 0) {
    const current_id = queue.shift();
    const current_layer = Layer.get(current_id);
    if (!current_layer) {
      continue;
    }

    let inferred_input_shape = null;
    if (external_source_shapes.has(current_id)) {
      inferred_input_shape = external_source_shapes.get(current_id);
    }

    for (let i = 0; i < current_layer.parent_ids.length; i++) {
      const parent_id = current_layer.parent_ids.get(i);
      const parent_output_shape = output_shapes.get(parent_id);
      if (!parent_output_shape) {
        continue;
      }

      if (!inferred_input_shape) {
        inferred_input_shape = parent_output_shape;
      } else if (!Tensor.shapes_equal(inferred_input_shape, parent_output_shape)) {
        inferred_input_shape = null;
        break;
      }
    }

    const output_shape = infer_layer_output_shape(current_layer, inferred_input_shape);
    if (Tensor.is_shape_valid(output_shape)) {
      output_shapes.set(current_id, output_shape);
    }

    for (let i = 0; i < current_layer.child_ids.length; i++) {
      const child_id = current_layer.child_ids[i];
      const new_degree = in_degree.get(child_id) - 1;
      in_degree.set(child_id, new_degree);

      if (new_degree === 0) {
        queue.push(child_id);
      }
    }
  }

  return { subnet_layers, subnet_layer_ids, output_shapes };
}

function get_training_output_layer_ids(subnet_layers, subnet_layer_ids) {
  const output_layer_ids = new Set();
  let has_loss_layers = false;

  for (let i = 0; i < subnet_layers.length; i++) {
    const id = subnet_layers[i];
    const layer = Layer.get(id);
    if (!layer || !Layer.is_loss(id)) {
      continue;
    }

    has_loss_layers = true;
    for (let parent_index = 0; parent_index < layer.parent_ids.length; parent_index++) {
      const parent_id = layer.parent_ids.get(parent_index);
      if (subnet_layer_ids.has(parent_id) && !Layer.is_loss(parent_id)) {
        output_layer_ids.add(parent_id);
      }
    }
  }

  if (has_loss_layers) {
    return Array.from(output_layer_ids);
  }

  for (let i = 0; i < subnet_layers.length; i++) {
    const id = subnet_layers[i];
    const layer = Layer.get(id);
    if (!layer || Layer.is_loss(id)) {
      continue;
    }

    if (layer.child_ids.length === 0) {
      output_layer_ids.add(id);
    }
  }

  return Array.from(output_layer_ids);
}

function get_training_validation_failure(subnet_entry) {
  const root_id = subnet_entry?.subnet_id;
  const root_layer = Layer.get(root_id);
  const first_sample = root_layer?.training_queue?.peek();

  if (!first_sample?.input?.shape || !first_sample?.target?.shape) {
    return null;
  }

  const target_dim = Tensor.last_dim(first_sample.target.shape);
  if (target_dim === null) {
    return null;
  }

  const { subnet_layers, subnet_layer_ids, output_shapes } = infer_subnet_output_shapes(
    root_id,
    first_sample.input.shape
  );
  const output_layer_ids = get_training_output_layer_ids(subnet_layers, subnet_layer_ids);

  for (let i = 0; i < output_layer_ids.length; i++) {
    const output_layer_id = output_layer_ids[i];
    const output_shape = output_shapes.get(output_layer_id);
    const output_dim = Tensor.last_dim(output_shape);

    if (output_dim === null || output_dim === target_dim) {
      continue;
    }

    const key = `${root_id}:${output_layer_id}:${output_dim}:${target_dim}`;
    return {
      key,
      root_id,
      output_layer_id,
      output_dim,
      target_dim,
      output_shape,
      target_shape: Tensor.normalize_shape(first_sample.target.shape),
      message:
        `Training paused: output dimension ${output_dim} does not match ` +
        `target dimension ${target_dim}.`,
      details:
        `Output shape ${Tensor.format_shape(output_shape)}, target shape ${Tensor.format_shape(first_sample.target.shape)}.`,
    };
  }

  return null;
}

export function get_mastermind_training_target_failures(mastermind) {
  if (!mastermind) {
    return [];
  }

  const failures = [];

  for (let i = 0; i < mastermind.subnets.length; i++) {
    const subnet_entry = mastermind.subnets.get(i);
    const failure = get_training_validation_failure(subnet_entry);
    if (!failure) {
      continue;
    }

    failures.push(failure);
  }

  return failures;
}

export function validate_mastermind_training_targets(mastermind) {
  return get_mastermind_training_target_failures(mastermind).length === 0;
}
