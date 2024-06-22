import Name from "@/utility/names.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";

export default class RenderPass {
    config = null;
    pass = null;

    constructor() { }

    static create(context, name, render_pass_config) {
        let render_pass = ResourceCache.get().fetch(CacheTypes.RENDER_PASS, Name.get_hash(name));
        if (!render_pass) {
            render_pass = new RenderPass();
            ResourceCache.get().store(CacheTypes.RENDER_PASS, Name.get_hash(name), render_pass);
        }
        render_pass.config = render_pass_config;
        return render_pass;
    }

    begin(context, encoder, pipeline) {
        this.pass = encoder.beginRenderPass(this.config);
        this.pass.setPipeline(pipeline);
    }

    end(context) {
        if (this.pass) {
            this.pass.end();
        }
    }
}