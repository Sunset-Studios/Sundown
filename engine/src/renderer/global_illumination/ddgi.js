import { SharedViewBuffer } from "../../core/shared_data.js";
import { RenderPassFlags } from "../renderer_types.js";
import { quat } from "gl-matrix";

export class GIProbeVolume {
  /**
   * @param {Array<number>} origin - [x, y, z] min corner of the volume in world space
   * @param {Array<number>} dims - [nx, ny, nz] number of probes along each axis
   * @param {Array<number>} volume_size - [sx, sy, sz] total size of the volume in world units
   * @param {number} probes_per_frame - number of probes to update each frame
   * @param {number} blend_factor - exponential moving average blend factor (0..1)
   */
  constructor(
    origin = [0, 0, 0],
    dims = [20, 20, 10],
    volume_size = [100, 100, 100],
    probes_per_frame = 16,
    blend_factor = 0.5
  ) {
    this.origin = origin;
    this.dims = dims;
    this.volume_size = volume_size;
    this.spacing = [
      volume_size[0] / dims[0],
      volume_size[1] / dims[1],
      volume_size[2] / dims[2],
    ];
    this.probe_count = dims[0] * dims[1] * dims[2];
    this.probes_per_frame = probes_per_frame;
    this.blend_factor = blend_factor;
    this.current_probe_index = 0;

    const probe_view_count = this.probes_per_frame * 6;
    this.probe_view_indices = new Uint32Array(probe_view_count);
    for (let i = 0; i < probe_view_count; ++i) {
      const view = SharedViewBuffer.add_view_data();
      //view.renderable_state = 1;
      view.occlusion_enabled = 0;
      this.probe_view_indices[i] = view.get_index();
    }
  }

  /**
   * Compute world-space position of a given probe index
   * @param {number} index
   * @returns {Array<number>} [x, y, z]
   */
  get_world_position_for_probe(index) {
    const nx = this.dims[0];
    const ny = this.dims[1];

    const rem = index % (nx * ny);
    const y = Math.floor(rem / nx);
    const x = rem % nx;
    const z = Math.floor(index / (nx * ny));

    return [
      this.origin[0] + x * this.spacing[0],
      this.origin[1] + y * this.spacing[1],
      this.origin[2] + z * this.spacing[2],
    ];
  }

  /** Reset origin */
  set_origin(origin) {
    this.origin = origin;
  }

  /**
   * Reset dimensions and recompute spacing and probe_count
   * @param {Array<number>} dims
   */
  set_dims(dims) {
    this.dims = dims;
    this._update_spacing_and_probe_count();
  }

  /**
   * Reset volume size and recompute spacing
   * @param {Array<number>} volume_size
   */
  set_volume_size(volume_size) {
    this.volume_size = volume_size;
    this._update_spacing_and_probe_count();
  }

  /**
   * Set all parameters at once
   * @param {Array<number>} origin
   * @param {Array<number>} dims
   * @param {Array<number>} volume_size
   * @param {number} probes_per_frame
   * @param {number} blend_factor
   */
  set_gi_probe_volume_params(origin, dims, volume_size, probes_per_frame, blend_factor) {
    if (origin) this.set_origin(origin);
    if (dims) this.set_dims(dims);
    if (volume_size) this.set_volume_size(volume_size);
    if (probes_per_frame !== undefined) this.probes_per_frame = probes_per_frame;
    if (blend_factor !== undefined) this.blend_factor = blend_factor;
  }

  /** Internal: recompute spacing and probe_count */
  _update_spacing_and_probe_count() {
    this.spacing = [
      this.volume_size[0] / this.dims[0],
      this.volume_size[1] / this.dims[1],
      this.volume_size[2] / this.dims[2],
    ];
    this.probe_count = this.dims[0] * this.dims[1] * this.dims[2];

    for (let i = 0; i < this.probe_view_indices.length; ++i) {
      SharedViewBuffer.remove_view_data(this.probe_view_indices[i], false);
    }

    this.probe_view_indices = new Uint32Array(this.probe_count);
    for (let i = 0; i < this.probe_count; ++i) {
      const view = SharedViewBuffer.add_view_data();
      //view.renderable_state = 1;
      view.occlusion_enabled = 0;
      this.probe_view_indices[i] = view.get_index();
    }
  }

  /**
   * Update and dispatch the DDGI probes via the selected strategy.
   * @param render_graph
   * @param resources { gi_params_buffer, gi_irradiance_image, gi_depth_image,
   *                    entity_transforms, entity_flags, compacted_object_instance_buffer, lights, probe_cubemap }
   */
  update(render_graph, resources) {
    // only 'raster' is supported currently
    GIProbeVolume._raster_update(this, render_graph, resources);
  }

  /**
   * Raster-based probe update: renders each probe's 6 faces and convolve.
   */
  static _raster_update(vol, render_graph, resources) {
    const { gi_params_buffer, gi_irradiance_image, gi_depth_image,
            entity_transforms, entity_flags,
            compacted_object_instance_buffer, lights, probe_cubemap } = resources;
    const probes = vol.probes_per_frame;
    const dims = vol.dims;
    for (let i = 0; i < probes; ++i) {
      const idx = vol.current_probe_index;
      const z = Math.floor(idx / (dims[0]*dims[1]));
      const rem = idx % (dims[0]*dims[1]);
      const y = Math.floor(rem / dims[0]);
      const x = rem % dims[0];

      // write GI parameters
      const buf = render_graph.get_buffer(gi_params_buffer);
      const o = vol.origin;
      const s = vol.spacing;
      buf.write([
        o[0] + x*s[0], o[1] + y*s[1], o[2] + z*s[2], 0,
        s[0], s[1], s[2], 0,
        dims[0], dims[1], dims[2], 0,
        idx, 0, 0, 0,
      ]);

      // prepare world position
      const worldPos = [o[0]+x*s[0], o[1]+y*s[1], o[2]+z*s[2], 1];
      const view_idx = vol.probe_view_indices[idx];
      const view_data = SharedViewBuffer.get_view_data(view_idx);
      view_data.view_position = worldPos;

      // 6 cubemap faces
      const dirs = [
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
      ];
      for (let face=0; face<6; ++face) {
        const dir = dirs[face];
        const q = quat.rotationTo(quat.create(), [0,0,1], dir);

        view_data.view_rotation = q;

        render_graph.add_pass(
          `ddgi_raster_${idx}_face_${face}`,
          RenderPassFlags.Graphics,
          {
            inputs: [entity_transforms, entity_flags, compacted_object_instance_buffer, lights],
            outputs: [{ name: probe_cubemap, array_layer: face }],
            shader_setup: vol.raster_setup,
          },
          (g, fd, encoder) => {
            const pass = g.get_physical_pass(fd.current_pass);
            pass.setViewport(0,0,vol.resolution,vol.resolution,0,1);
            pass.setScissorRect(0,0,vol.resolution,vol.resolution);
            MeshTaskQueue.draw_scene_direct(pass);
          }
        );
      }

      // convolve compute pass
      render_graph.add_pass(
        `ddgi_accum_${idx}`,
        RenderPassFlags.Compute,
        {
          shader_setup: { pipeline_shaders:{ compute:{ path: vol.accum_shader } } },
          inputs: [probe_cubemap, gi_params_buffer],
          outputs: [gi_irradiance_image, gi_depth_image],
        },
        (g, fd, encoder) => {
          const pass = g.get_physical_pass(fd.current_pass);
          pass.dispatch(1,1,1);
        }
      );
    }
  }

  /**
   * Stub for future ray-traced updates
   */
  static _raytrace_update(vol, render_graph, resources) {
    throw new Error("RayTrace update not implemented");
  }
}
