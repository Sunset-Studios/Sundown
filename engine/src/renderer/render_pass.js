import { Name } from "../utility/names.js";
import { ResourceCache, CacheTypes } from "./resource_cache.js";

/**
 * Flags for render passes in the render graph.
 * @enum {number}
 */
export const RenderPassFlags = Object.freeze({
  /** No flags */
  None: 0,
  /** Indicates a graphics pass */
  Graphics: 1,
  /** Indicates a present pass */
  Present: 2,
  /** Indicates a compute pass */
  Compute: 4,
  /** Indicates a graph-local pass */
  GraphLocal: 8,
});

export class RenderPass {
  pass = null;
  config = null;

  init(config) {
    this.config = config;
  }

  begin(encoder, pipeline) {
    if (this.config.flags & RenderPassFlags.Graphics) {
      const attachments = this.config.attachments
        .map((attachment) => {
          const image = ResourceCache.get().fetch(
            CacheTypes.IMAGE,
            attachment.image
          );
          return {
            view: image.get_view(attachment.view_index) || image.view,
            clearValue: image.config.clear_value ?? { r: 0, g: 0, b: 0, a: 1 },
            loadOp: image.config.load_op ?? 'clear',
            storeOp: image.config.store_op ?? 'store',
          };
        });

        const pass_desc = {
          label: this.config.name,
          colorAttachments: attachments,
        };

      if (this.config.depth_stencil_attachment) {
        const depth_stencil_image = ResourceCache.get().fetch(
          CacheTypes.IMAGE,
          this.config.depth_stencil_attachment.image
        );
        pass_desc.depthStencilAttachment = {
          view: depth_stencil_image.get_view(this.config.depth_stencil_attachment.view_index) || depth_stencil_image.view,
          depthClearValue: depth_stencil_image.config.clear_value ?? 0.0,
          depthLoadOp: depth_stencil_image.config.load_op ?? "load",
          depthStoreOp: depth_stencil_image.config.store_op ?? "store",
        };
      }

      this.pass = encoder.beginRenderPass(pass_desc);
    } else if (this.config.flags & RenderPassFlags.Compute) {
      this.pass = encoder.beginComputePass({
        label: this.config.name,
      });
    }
    
    if (this.config.viewport) {
      this.pass.setViewport(this.config.viewport);
    }
    if (this.config.scissor_rect) {
      this.pass.setScissorRect(this.config.scissor_rect);
    }
    if (this.config.vertex_buffer) {
      this.pass.setVertexBuffer(this.config.vertex_buffer);
    }
    if (this.config.index_buffer) {
      this.pass.setIndexBuffer(this.config.index_buffer);
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

  dispatch(x, y, z) {
    if (this.config.flags & RenderPassFlags.Compute) {
      this.pass.dispatchWorkgroups(x, y, z);
    }
  }

  end() {
    if (this.pass) {
      this.pass.end();
    }
  }

  static create(config) {
    let render_pass = ResourceCache.get().fetch(
      CacheTypes.PASS,
      Name.from(config.name)
    );
    if (!render_pass) {
      render_pass = new RenderPass();
      render_pass.init(config);
      ResourceCache.get().store(
        CacheTypes.PASS,
        Name.from(config.name),
        render_pass
      );
    }
    return render_pass;
  }
}
