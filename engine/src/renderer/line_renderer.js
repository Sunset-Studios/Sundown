import { Renderer } from "./renderer.js";
import { RandomAccessAllocator } from "../memory/allocator.js";
import { Buffer } from "./buffer.js";
import { vec3, vec4 } from "gl-matrix";

// Constants
const LINE_WIDTH = 0.05;
const DEFAULT_LINE_COLOR = [1.0, 1.0, 1.0, 1.0]; // White
const DEFAULT_COLLECTION_ID = 0; // Default collection ID

class LinePosition {
  start = [0.0, 0.0, 0.0, 1.0];
  end = [0.0, 0.0, 0.0, 1.0];
  collection_id = DEFAULT_COLLECTION_ID; // Collection ID for grouping lines
  active = true; // Whether this line is active or has been "deleted"
}

/**
 * A debug renderer for drawing lines in world space
 */
export class LineRenderer {
  static enabled = true;
  static line_positions = new RandomAccessAllocator(1024, LinePosition);
  static line_data = new RandomAccessAllocator(1024, [0.0, 0.0, 0.0, LINE_WIDTH]);
  static position_buffer = null;
  static line_data_buffer = null;
  static transform_buffer = null;
  static gpu_data_dirty = false;

  // Collection management
  static current_collection_id = DEFAULT_COLLECTION_ID;
  static next_collection_id = 1;
  static active_collections = new Set([DEFAULT_COLLECTION_ID]);
  static collection_line_counts = new Map([[DEFAULT_COLLECTION_ID, 0]]);
  static last_visible_line_count = 0;

  // Free list management
  static free_indices = []; // Indices of "deleted" lines that can be reused
  static active_line_count = 0; // Total number of active lines

  /**
   * Start a new line collection
   * @returns {number} The ID of the new collection
   */
  static start_collection() {
    const collection_id = this.next_collection_id++;
    this.current_collection_id = collection_id;
    this.active_collections.add(collection_id);
    this.collection_line_counts.set(collection_id, 0);
    return collection_id;
  }

  /**
   * End the current collection and return to the default collection
   */
  static end_collection() {
    this.current_collection_id = DEFAULT_COLLECTION_ID;
  }

  /**
   * Set the active collection ID
   * @param {number} collection_id - The collection ID to set as active
   */
  static set_active_collection(collection_id) {
    if (this.active_collections.has(collection_id)) {
      this.current_collection_id = collection_id;
    } else {
      console.warn(`Collection ID ${collection_id} does not exist`);
    }
  }

  /**
   * Show a specific collection
   * @param {number} collection_id - The collection ID to show
   */
  static show_collection(collection_id) {
    if (this.collection_line_counts.has(collection_id)) {
      this.active_collections.add(collection_id);
      this.gpu_data_dirty = true;
    }
  }

  /**
   * Hide a specific collection
   * @param {number} collection_id - The collection ID to hide
   */
  static hide_collection(collection_id) {
    if (collection_id !== DEFAULT_COLLECTION_ID && this.active_collections.has(collection_id)) {
      this.active_collections.delete(collection_id);
      this.gpu_data_dirty = true;
    }
  }

  /**
   * Clear a specific collection
   * @param {number} collection_id - The collection ID to clear
   */
  static clear_collection(collection_id) {
    if (!this.collection_line_counts.has(collection_id)) {
      return;
    }

    // Mark all lines in this collection as inactive and add their indices to the free list
    for (let i = 0; i < this.line_positions.length; i++) {
      const position = this.line_positions.get(i);
      if (position.collection_id === collection_id && position.active) {
        position.active = false; // Mark as inactive
        this.free_indices.push(i); // Add to free list
        this.active_line_count--;
      }
    }

    // Update collection line count
    this.collection_line_counts.set(collection_id, 0);

    this.gpu_data_dirty = true;
  }

  /**
   * Get all active collection IDs
   * @returns {Array} Array of active collection IDs
   */
  static get_active_collections() {
    return Array.from(this.active_collections);
  }

  /**
   * Add a line to be drawn
   * @param {Array} start - Start position [x, y, z]
   * @param {Array} end - End position [x, y, z]
   * @param {Array} color - Line color [r, g, b]
   * @param {number} width - Line width
   * @param {number} collection_id - Optional collection ID (uses current collection if not specified)
   */
  static add_line(
    start,
    end,
    color = DEFAULT_LINE_COLOR,
    width = LINE_WIDTH,
    collection_id = null
  ) {
    let line_position;
    let line_data;
    let index;

    // Try to reuse a free slot if available
    if (this.free_indices.length > 0) {
      index = this.free_indices.pop();
      line_position = this.line_positions.get(index);
      line_data = this.line_data.get(index);
      line_position.active = true; // Mark as active
    } else {
      // Allocate new if no free slots
      line_position = this.line_positions.allocate();
      line_data = this.line_data.allocate();
      index = this.line_positions.length - 1;
    }

    line_position.start[0] = start[0];
    line_position.start[1] = start[1];
    line_position.start[2] = start[2];
    line_position.start[3] = 1.0;
    line_position.end[0] = end[0];
    line_position.end[1] = end[1];
    line_position.end[2] = end[2];
    line_position.end[3] = 1.0;

    // Set collection ID (use current collection if not specified)
    const actual_collection_id =
      collection_id !== null ? collection_id : this.current_collection_id;
    line_position.collection_id = actual_collection_id;
    // Update collection line count
    const current_count = this.collection_line_counts.get(actual_collection_id) || 0;
    this.collection_line_counts.set(actual_collection_id, current_count + 1);

    // Set line color and width
    line_data[0] = color[0];
    line_data[1] = color[1];
    line_data[2] = color[2];
    line_data[3] = width;

    this.active_line_count++;
    this.gpu_data_dirty = true;
  }

  /**
   * Add a box outline
   * @param {Array} min_point - Minimum corner [x, y, z]
   * @param {Array} max_point - Maximum corner [x, y, z]
   * @param {Array} color - Line color [r, g, b]
   * @param {number} width - Line width
   * @param {number} collection_id - Optional collection ID (uses current collection if not specified)
   */
  static #box_edges = [
    [0, 1],
    [1, 3],
    [3, 2],
    [2, 0], // Bottom face
    [4, 5],
    [5, 7],
    [7, 6],
    [6, 4], // Top face
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7], // Connecting edges
  ];
  static #corners = [
    [0, 0, 0], // Bottom face
    [1, 0, 0], // Top face
    [0, 1, 0], // Connecting edges
    [1, 1, 0], // Connecting edges
    [0, 0, 1], // Connecting edges
    [1, 0, 1], // Connecting edges
    [0, 1, 1], // Connecting edges
    [1, 1, 1], // Connecting edges
  ];
  static add_box(
    min_point,
    max_point,
    color = DEFAULT_LINE_COLOR,
    width = LINE_WIDTH,
    collection_id = null
  ) {
    // Define the 8 corners of the box
    this.#corners[0][0] = min_point[0];
    this.#corners[0][1] = min_point[1];
    this.#corners[0][2] = min_point[2];
    this.#corners[1][0] = max_point[0];
    this.#corners[1][1] = min_point[1];
    this.#corners[1][2] = min_point[2];
    this.#corners[2][0] = min_point[0];
    this.#corners[2][1] = max_point[1];
    this.#corners[2][2] = min_point[2];
    this.#corners[3][0] = max_point[0];
    this.#corners[3][1] = max_point[1];
    this.#corners[3][2] = min_point[2];
    this.#corners[4][0] = min_point[0];
    this.#corners[4][1] = min_point[1];
    this.#corners[4][2] = max_point[2];
    this.#corners[5][0] = max_point[0];
    this.#corners[5][1] = min_point[1];
    this.#corners[5][2] = max_point[2];
    this.#corners[6][0] = min_point[0];
    this.#corners[6][1] = max_point[1];
    this.#corners[6][2] = max_point[2];
    this.#corners[7][0] = max_point[0];
    this.#corners[7][1] = max_point[1];
    this.#corners[7][2] = max_point[2];
    // Create a line for each edge
    for (const [start_idx, end_idx] of this.#box_edges) {
      this.add_line(this.#corners[start_idx], this.#corners[end_idx], color, width, collection_id);
    }
  }

  /**
   * Add a grid of lines
   * @param {number} size - Grid size
   * @param {number} spacing - Grid spacing
   * @param {Array} center - Grid center [x, y, z]
   * @param {Array} color - Line color [r, g, b]
   * @param {number} width - Line width
   * @param {number} collection_id - Optional collection ID (uses current collection if not specified)
   */
  static add_grid(
    size = 10,
    spacing = 1,
    center = [0, 0, 0],
    color = DEFAULT_LINE_COLOR,
    width = LINE_WIDTH,
    collection_id = null
  ) {
    const half_size = (size * spacing) / 2;

    // Add horizontal lines (along X axis)
    for (let i = 0; i <= size; i++) {
      const z = center[2] - half_size + i * spacing;
      this.add_line(
        [center[0] - half_size, center[1], z],
        [center[0] + half_size, center[1], z],
        color,
        width,
        collection_id
      );
    }

    // Add vertical lines (along Z axis)
    for (let i = 0; i <= size; i++) {
      const x = center[0] - half_size + i * spacing;
      this.add_line(
        [x, center[1], center[2] - half_size],
        [x, center[1], center[2] + half_size],
        color,
        width,
        collection_id
      );
    }
  }

  /**
   * Add a circle outline
   * @param {Array} center - Circle center [x, y, z]
   * @param {number} radius - Circle radius
   * @param {Array} normal - Circle normal vector [x, y, z]
   * @param {number} segments - Number of segments
   * @param {Array} color - Line color [r, g, b]
   * @param {number} width - Line width
   * @param {number} collection_id - Optional collection ID (uses current collection if not specified)
   */
  static add_circle(
    center,
    radius,
    normal = [0, 1, 0],
    segments = 32,
    color = DEFAULT_LINE_COLOR,
    width = LINE_WIDTH,
    collection_id = null
  ) {
    // Normalize the normal vector
    const norm = vec3.normalize(vec3.create(), normal);

    // Create two perpendicular vectors in the plane
    let tangent = vec3.create();
    if (Math.abs(norm[0]) < Math.abs(norm[1])) {
      vec3.set(tangent, 1, 0, 0);
    } else {
      vec3.set(tangent, 0, 1, 0);
    }
    vec3.cross(tangent, norm, tangent);
    vec3.normalize(tangent, tangent);

    const bitangent = vec3.cross(vec3.create(), norm, tangent);

    // Generate points around the circle
    const points = [];
    for (let i = 0; i < segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      const x =
        center[0] + radius * (Math.cos(angle) * tangent[0] + Math.sin(angle) * bitangent[0]);
      const y =
        center[1] + radius * (Math.cos(angle) * tangent[1] + Math.sin(angle) * bitangent[1]);
      const z =
        center[2] + radius * (Math.cos(angle) * tangent[2] + Math.sin(angle) * bitangent[2]);
      points.push([x, y, z]);
    }

    // Connect the points with lines
    for (let i = 0; i < segments; i++) {
      const next = (i + 1) % segments;
      this.add_line(points[i], points[next], color, width, collection_id);
    }
  }

  /**
   * Add a sphere wireframe
   * @param {Array} center - Sphere center [x, y, z]
   * @param {number} radius - Sphere radius
   * @param {number} rings - Number of horizontal rings
   * @param {number} segments - Number of segments per ring
   * @param {Array} color - Line color [r, g, b]
   * @param {number} width - Line width
   * @param {number} collection_id - Optional collection ID (uses current collection if not specified)
   */
  static add_sphere(
    center,
    radius,
    rings = 8,
    segments = 16,
    color = DEFAULT_LINE_COLOR,
    width = LINE_WIDTH,
    collection_id = null
  ) {
    // Add horizontal rings
    for (let i = 0; i <= rings; i++) {
      const phi = (i / rings) * Math.PI;
      const ring_radius = radius * Math.sin(phi);
      const y = center[1] + radius * Math.cos(phi);

      this.add_circle(
        [center[0], y, center[2]],
        ring_radius,
        [0, 1, 0],
        segments,
        color,
        width,
        collection_id
      );
    }

    // Add vertical semicircles
    for (let i = 0; i < segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      const x_dir = Math.cos(theta);
      const z_dir = Math.sin(theta);

      // Create points for the semicircle
      const points = [];
      for (let j = 0; j <= rings * 2; j++) {
        const phi = (j / (rings * 2)) * Math.PI;
        const x = center[0] + radius * Math.sin(phi) * x_dir;
        const y = center[1] + radius * Math.cos(phi);
        const z = center[2] + radius * Math.sin(phi) * z_dir;
        points.push([x, y, z]);
      }

      // Connect the points with lines
      for (let j = 0; j < points.length - 1; j++) {
        this.add_line(points[j], points[j + 1], color, width, collection_id);
      }
    }
  }

  /**
   * Add an arrow
   * @param {Array} start - Start position [x, y, z]
   * @param {Array} end - End position [x, y, z]
   * @param {number} head_size - Size of the arrow head
   * @param {Array} color - Line color [r, g, b]
   * @param {number} width - Line width
   * @param {number} collection_id - Optional collection ID (uses current collection if not specified)
   */
  static add_arrow(
    start,
    end,
    head_size = 0.2,
    color = DEFAULT_LINE_COLOR,
    width = LINE_WIDTH,
    collection_id = null
  ) {
    // Draw the main line
    this.add_line(start, end, color, width, collection_id);

    // Calculate direction vector
    const dir = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];

    // Normalize direction
    const length = Math.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    if (length < 0.0001) return; // Too short to draw an arrow

    const norm_dir = [dir[0] / length, dir[1] / length, dir[2] / length];

    // Find perpendicular vectors
    let perp1 = [0, 1, 0];
    if (Math.abs(norm_dir[1]) > 0.9) {
      perp1 = [1, 0, 0];
    }

    const perp2 = [
      norm_dir[1] * perp1[2] - norm_dir[2] * perp1[1],
      norm_dir[2] * perp1[0] - norm_dir[0] * perp1[2],
      norm_dir[0] * perp1[1] - norm_dir[1] * perp1[0],
    ];

    const perp1_length = Math.sqrt(perp2[0] * perp2[0] + perp2[1] * perp2[1] + perp2[2] * perp2[2]);
    perp1[0] = perp2[0] / perp1_length;
    perp1[1] = perp2[1] / perp1_length;
    perp1[2] = perp2[2] / perp1_length;

    const perp3 = [
      norm_dir[1] * perp1[2] - norm_dir[2] * perp1[1],
      norm_dir[2] * perp1[0] - norm_dir[0] * perp1[2],
      norm_dir[0] * perp1[1] - norm_dir[1] * perp1[0],
    ];

    // Calculate arrow head points
    const arrow_base = [
      end[0] - norm_dir[0] * head_size,
      end[1] - norm_dir[1] * head_size,
      end[2] - norm_dir[2] * head_size,
    ];

    // Create 4 points around the base of the arrow head
    const arrow_points = [];
    for (let i = 0; i < 4; i++) {
      const angle = (i / 4) * Math.PI * 2;
      const x =
        arrow_base[0] + head_size * 0.5 * (Math.cos(angle) * perp1[0] + Math.sin(angle) * perp3[0]);
      const y =
        arrow_base[1] + head_size * 0.5 * (Math.cos(angle) * perp1[1] + Math.sin(angle) * perp3[1]);
      const z =
        arrow_base[2] + head_size * 0.5 * (Math.cos(angle) * perp1[2] + Math.sin(angle) * perp3[2]);
      arrow_points.push([x, y, z]);
    }

    // Draw lines from arrow head to base points
    for (const point of arrow_points) {
      this.add_line(end, point, color, width, collection_id);
    }

    // Connect the base points
    for (let i = 0; i < arrow_points.length; i++) {
      const next = (i + 1) % arrow_points.length;
      this.add_line(arrow_points[i], arrow_points[next], color, width, collection_id);
    }
  }

  /**
   * Add a coordinate axes visualization
   * @param {Array} origin - Origin position [x, y, z]
   * @param {number} size - Size of the axes
   * @param {number} collection_id - Optional collection ID (uses current collection if not specified)
   */
  static add_axes(origin, size = 1.0, collection_id = null) {
    // X axis (red)
    this.add_arrow(
      origin,
      [origin[0] + size, origin[1], origin[2]],
      size * 0.2,
      [1, 0, 0, 1],
      LINE_WIDTH,
      collection_id
    );

    // Y axis (green)
    this.add_arrow(
      origin,
      [origin[0], origin[1] + size, origin[2]],
      size * 0.2,
      [0, 1, 0, 1],
      LINE_WIDTH,
      collection_id
    );

    // Z axis (blue)
    this.add_arrow(
      origin,
      [origin[0], origin[1], origin[2] + size],
      size * 0.2,
      [0, 0, 1, 1],
      LINE_WIDTH,
      collection_id
    );
  }

  /**
   * Clear all lines
   */
  static clear() {
    // Reset allocators
    this.line_positions.reset();
    this.line_data.reset();

    // Reset collection management
    this.active_collections = new Set([DEFAULT_COLLECTION_ID]);
    this.collection_line_counts = new Map([[DEFAULT_COLLECTION_ID, 0]]);
    this.current_collection_id = DEFAULT_COLLECTION_ID;

    // Reset free list management
    this.free_indices = [];
    this.active_line_count = 0;

    this.gpu_data_dirty = true;
  }

  /**
   * Enable or disable the renderer
   */
  static toggle() {
    this.enabled = !this.enabled;
  }

  /**
   * Update GPU buffers with current line data
   */
  static to_gpu_data() {
    if (!this.gpu_data_dirty || this.line_positions.length === 0) {
      return {
        position_buffer: this.position_buffer,
        line_data_buffer: this.line_data_buffer,
        transform_buffer: this.transform_buffer,
        visible_line_count: this.last_visible_line_count,
      };
    }

    // Count visible lines (only from active collections and active lines)
    this.last_visible_line_count = 0;
    for (let i = 0; i < this.line_positions.length; i++) {
      const position = this.line_positions.get(i);
      if (position.active && this.active_collections.has(position.collection_id)) {
        this.last_visible_line_count++;
      }
    }

    // Prepare data for transform buffer
    const position_data = new Float32Array(this.last_visible_line_count * 8);
    const line_data = new Float32Array(this.last_visible_line_count * 4);

    let visible_index = 0;
    for (let i = 0; i < this.line_positions.length; i++) {
      const position = this.line_positions.get(i);

      // Skip lines from inactive collections or inactive lines
      if (!position.active || !this.active_collections.has(position.collection_id)) {
        continue;
      }

      const pos_offset = visible_index * 8;
      position_data[pos_offset] = position.start[0];
      position_data[pos_offset + 1] = position.start[1];
      position_data[pos_offset + 2] = position.start[2];
      position_data[pos_offset + 3] = position.start[3];
      position_data[pos_offset + 4] = position.end[0];
      position_data[pos_offset + 5] = position.end[1];
      position_data[pos_offset + 6] = position.end[2];
      position_data[pos_offset + 7] = position.end[3];

      const data_offset = visible_index * 4;
      const data = this.line_data.get(i);
      line_data[data_offset] = data[0]; // R
      line_data[data_offset + 1] = data[1]; // G
      line_data[data_offset + 2] = data[2]; // B
      line_data[data_offset + 3] = data[3]; // Width

      visible_index++;
    }

    // Create position buffer
    if (!this.position_buffer || this.position_buffer.config.size < position_data.byteLength) {
      this.position_buffer = Buffer.create({
        name: "line_position_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        raw_data: position_data,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    } else {
      this.position_buffer.write(position_data);
    }

    // Create line data buffer
    if (!this.line_data_buffer || this.line_data_buffer.config.size < line_data.byteLength) {
      this.line_data_buffer = Buffer.create({
        name: "line_data_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        raw_data: line_data,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    } else {
      this.line_data_buffer.write(line_data);
    }

    // Create transform buffer
    if (
      !this.transform_buffer ||
      this.transform_buffer.config.size < this.last_visible_line_count * 16
    ) {
      this.transform_buffer = Buffer.create({
        name: "line_transform_buffer",
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        size: this.last_visible_line_count * 16,
        force: true,
      });
      Renderer.get().mark_bind_groups_dirty(true);
    }

    this.gpu_data_dirty = false;

    return {
      position_buffer: this.position_buffer,
      line_data_buffer: this.line_data_buffer,
      transform_buffer: this.transform_buffer,
      visible_line_count: this.last_visible_line_count,
    };
  }
}
