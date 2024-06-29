import Name from "@/utility/names.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";

export class PipelineState {
    pipeline = null;
    
    init_render_pipeline(context, name, config) {
        this.pipeline = context.device.createRenderPipeline({
            label: Name.string(name),
            layout: 'auto',
            ...config
        });
    }

    init_compute_pipeline(context, name, config) {
        this.pipeline = context.device.createComputePipeline({
            label: Name.string(name),
            layout: 'auto',
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