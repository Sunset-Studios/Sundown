import { Name } from "../utility/names.js";
import { ResourceCache } from "./resource_cache.js";
import { RenderPassFlags, CacheTypes, BindGroupType } from "./renderer_types.js";
import { GPUTimeQuery } from "./query.js";
import { TypedVector } from "../memory/container.js";

export class RenderPass {
  static all_passes = new TypedVector(256, 0, BigInt64Array);

  pass = null;
  config = null;
  frame_attachments = [];
  frame_bind_groups = Array(BindGroupType.Num).fill(null);
  timer_query_indices = [0, 0];

  init(config) {
    this.config = config;
  }

  begin(encoder, pipeline) {
    if (this.config.flags & RenderPassFlags.Graphics) {
      const attachments = this.config.attachments.map((attachment) => {
        const image = ResourceCache.get().fetch(CacheTypes.IMAGE, attachment.image);
        return {
          view: image.get_view(attachment.view_index) || image.view,
          clearValue: image.config.clear_value ?? { r: 0, g: 0, b: 0, a: 1 },
          loadOp: image.config.load_op ?? "clear",
          storeOp: image.config.store_op ?? "store",
        };
      });

      let pass_desc = {
        label: this.config.name,
        colorAttachments: attachments,
      };

      if (__DEV__ && GPUTimeQuery.query_set) {
        this.timer_query_indices[0] = GPUTimeQuery.allocate();
        this.timer_query_indices[1] = GPUTimeQuery.allocate();
        pass_desc.timestampWrites = {
          querySet: GPUTimeQuery.query_set,
          beginningOfPassWriteIndex: this.timer_query_indices[0],
          endOfPassWriteIndex: this.timer_query_indices[1],
        };
      }

      if (this.config.depth_stencil_attachment) {
        const depth_stencil_image = ResourceCache.get().fetch(
          CacheTypes.IMAGE,
          this.config.depth_stencil_attachment.image
        );
        pass_desc.depthStencilAttachment = {
          view:
            depth_stencil_image.get_view(this.config.depth_stencil_attachment.view_index) ||
            depth_stencil_image.view,
          depthClearValue: depth_stencil_image.config.clear_value ?? 0.0,
          depthLoadOp: depth_stencil_image.config.load_op ?? "load",
          depthStoreOp: depth_stencil_image.config.store_op ?? "store",
        };
      }

      this.pass = encoder.beginRenderPass(pass_desc);
    } else if (this.config.flags & RenderPassFlags.Compute) {
      const compute_pass_desc = { label: this.config.name };

      if (__DEV__ && GPUTimeQuery.query_set) {
        this.timer_query_indices[0] = GPUTimeQuery.allocate();
        this.timer_query_indices[1] = GPUTimeQuery.allocate();
        compute_pass_desc.timestampWrites = {
          querySet: GPUTimeQuery.query_set,
          beginningOfPassWriteIndex: this.timer_query_indices[0],
          endOfPassWriteIndex: this.timer_query_indices[1],
        };
      }

      this.pass = encoder.beginComputePass(compute_pass_desc);
    }

    if (this.config.viewport) {
      this.pass.setViewport(
        this.config.viewport.x,
        this.config.viewport.y,
        this.config.viewport.width,
        this.config.viewport.height,
        this.config.viewport.min_depth,
        this.config.viewport.max_depth
      );
    }
    if (this.config.scissor_rect) {
      this.pass.setScissorRect(
        this.config.scissor_rect.x,
        this.config.scissor_rect.y,
        this.config.scissor_rect.width,
        this.config.scissor_rect.height
      );
    }
    if (this.config.vertex_buffer) {
      this.pass.setVertexBuffer(this.config.vertex_buffer);
    }
    if (this.config.index_buffer) {
      this.pass.setIndexBuffer(this.config.index_buffer, this.config.index_buffer.element_type);
    }

    if (pipeline) {
      this.set_pipeline(pipeline);
    }
  }

  set_pipeline(pipeline) {
    this.pass.setPipeline(pipeline.pipeline);
  }

  set_attachments(attachments) {
    this.config.attachments = attachments;
  }

  set_depth_stencil_attachment(attachment) {
    this.config.depth_stencil_attachment = attachment;
  }

  set_viewport(viewport) {
    this.config.viewport = viewport;
  }

  set_scissor_rect(scissor_rect) {
    this.config.scissor_rect = scissor_rect;
  }

  dispatch(x, y, z) {
    if (this.config.flags & RenderPassFlags.Compute) {
      this.pass.dispatchWorkgroups(x, y, z);
    }
  }

  dispatch_indirect(buffer, offset = 0) {
    if (this.config.flags & RenderPassFlags.Compute) {
      this.pass.dispatchWorkgroupsIndirect(buffer.buffer, offset);
    }
  }

  end() {
    if (this.pass) {
      this.pass.end();
    }
  }

  static create(config) {
    let name_hash = Name.from(config.name);
    let render_pass = ResourceCache.get().fetch(CacheTypes.PASS, name_hash);
    if (!render_pass) {
      render_pass = new RenderPass();
      render_pass.init(config);
      ResourceCache.get().store(CacheTypes.PASS, name_hash, render_pass);
      RenderPass.all_passes.push(BigInt(name_hash));
    }
    return render_pass;
  }
}
