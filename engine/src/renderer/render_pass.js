import Name from "@/utility/names.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";

export const RenderPassType = Object.freeze({
    Graphics: 0,
    Compute: 1,
});

/**
 * Flags for render passes in the render graph.
 * @enum {number}
 */
export const RenderPassFlags = Object.freeze({
  /** No flags */
  None: 0,
  /** Indicates a present pass */
  Present: 1,
  /** Indicates a compute pass */
  Compute: 2,
  /** Indicates a graph-local pass */
  GraphLocal: 4,
});

export class RenderPass {
    pass = null;
    config = null;
    type = null;

    init(type, config) {
        this.type = type;
        this.config = config;
    }

    begin(encoder, pipeline) {
        if (this.type === RenderPassType.Graphics) {
            const attachments = this.config.attachments.map((attachment) => {
                const image = ResourceCache.get().fetch(CacheTypes.IMAGE, attachment.image);
                return {
                    view: image.image.createView(),
                    clearValue: image.config.clear_value,
                    loadOp: image.config.load_op,
                    storeOp: image.config.store_op,
                };
            });

            this.pass = encoder.beginRenderPass({
                label: this.config.name,
                colorAttachments: attachments,
                depthStencilAttachment: this.config.depth_stencil_attachment,
            });
        } else if (this.type === RenderPassType.Compute) {
            this.pass = encoder.beginComputePass({
                label: this.config.name,
            });
        }

        this.pass.setPipeline(pipeline.pipeline);

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
    }

    dispatch(x, y, z) {
        if (this.type === RenderPassType.Compute) {
            this.pass.dispatchWorkgroups(x, y, z);
        }
    }

    end() {
        if (this.pass) {
            this.pass.end();
        }
    }

    static create(type, config) {
        let render_pass = ResourceCache.get().fetch(CacheTypes.PASS, Name.from(config.name));
        if (!render_pass) {
            render_pass = new RenderPass();
            render_pass.init(type, config);
            ResourceCache.get().store(CacheTypes.PASS, Name.from(config.name), render_pass);
        }
        return render_pass;
    }
}