import { MeshData } from "./mesh_data";
import { Mesh } from "./mesh";

/**
 * Submit a draw quad command to the given render pass
 * 
 * @param {*} render_pass The render pass object to use when scheduling the quad draw
 * @param {*} instance_count The number of quads to instance in this draw
 * @param {*} first_instance The first instance to draw
 */
export function draw_quad(render_pass, instance_count = 1, first_instance = 0) {
  const index_buffer = MeshData.index_buffer;
  const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;

  const mesh = Mesh.quad();
  render_pass.pass.setIndexBuffer(
    index_buffer.buffer,
    index_buffer.config.element_type,
    mesh.index_buffer_offset * index_buffer_multiplier,
    mesh.index_count * index_buffer_multiplier
  );
  render_pass.pass.drawIndexed(mesh.index_count, instance_count, 0, mesh.vertex_buffer_offset, first_instance);
}

/**
 * Submit a draw cube command to the given render pass
 * 
 * @param {*} render_pass The render pass object to use when scheduling the cube draw
 * @param {*} instance_count The number of cubes to instance in this draw
 * @param {*} first_instance The first instance to draw
 */
export function draw_cube(render_pass, instance_count = 1, first_instance = 0) {
  const index_buffer = MeshData.index_buffer;
  const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;

  const mesh = Mesh.cube();
  render_pass.pass.setIndexBuffer(
    index_buffer.buffer,
    index_buffer.config.element_type,
    mesh.index_buffer_offset * index_buffer_multiplier,
    mesh.index_count * index_buffer_multiplier
  );
  render_pass.pass.drawIndexed(mesh.index_count, instance_count, 0, mesh.vertex_buffer_offset, first_instance);
}

/**
 * Submit a draw sphere command to the given render pass
 * 
 * @param {*} render_pass The render pass object to use when scheduling the sphere draw
 * @param {*} instance_count The number of spheres to instance in this draw
 * @param {*} first_instance The first instance to draw
 */
export function draw_sphere(render_pass, instance_count = 1, first_instance = 0) {
  const index_buffer = MeshData.index_buffer;
  const index_buffer_multiplier = index_buffer.config.element_type === "uint16" ? 2 : 4;

  const mesh = Mesh.sphere();
  render_pass.pass.setIndexBuffer(
    index_buffer.buffer,
    index_buffer.config.element_type,
    mesh.index_buffer_offset * index_buffer_multiplier,
    mesh.index_count * index_buffer_multiplier
  );
  render_pass.pass.drawIndexed(mesh.index_count, instance_count, 0, mesh.vertex_buffer_offset, first_instance);
}