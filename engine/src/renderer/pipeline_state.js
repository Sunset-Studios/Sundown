import { Name } from "../utility/names.js";
import { ResourceCache, CacheTypes } from "./resource_cache.js";

export class PipelineState {
    pipeline = null;
    layout = null;
    
    init_render_pipeline(context, name, config) {
        this.layout = context.device.createPipelineLayout({
            label: Name.string(name),
            bindGroupLayouts: config.bind_layouts,
        });
        this.pipeline = context.device.createRenderPipeline({
            label: Name.string(name),
            layout: this.layout,
            ...config
        });
    }

    init_compute_pipeline(context, name, config) {
        this.layout = context.device.createPipelineLayout({
            label: Name.string(name),
            bindGroupLayouts: config.bind_layouts,
        });
        this.pipeline = context.device.createComputePipeline({
            label: Name.string(name),
            layout: this.layout,
            ...config
        });
    }

    static create_render(context, name, config) {
        let name_hash = Name.from(name);
        let pipeline_state = ResourceCache.get().fetch(CacheTypes.PIPELINE_STATE, name_hash);
        if (!pipeline_state) {
            pipeline_state = new PipelineState();
            pipeline_state.init_render_pipeline(context, name_hash, config);
            ResourceCache.get().store(CacheTypes.PIPELINE_STATE, name_hash, pipeline_state);
        }
        return pipeline_state;
    }

    static create_compute(context, name, config) {
        let name_hash = Name.from(name);
        let pipeline_state = ResourceCache.get().fetch(CacheTypes.PIPELINE_STATE, name_hash);
        if (!pipeline_state) {
            pipeline_state = new PipelineState();
            pipeline_state.init_compute_pipeline(context, name_hash, config);
            ResourceCache.get().store(CacheTypes.PIPELINE_STATE, name_hash, pipeline_state);
        }
        return pipeline_state;
    }
}