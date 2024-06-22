import Name from "@/utility/names.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";

class PipelineState {
    pipeline = null;
    
    constructor() { }

    init_render_pipeline(context, name, config) {
        this.pipeline = context.device.createRenderPipeline({
            label: Name.get_string(name),
            layout: 'auto',
            ...config
        });
    }

    init_compute_pipeline(context, name, config) {
        this.pipeline = context.device.createComputePipeline({
            label: Name.get_string(name),
            layout: 'auto',
            ...config
        });
    }

    static create_render(context, name, config) {
        let pipeline_state = ResourceCache.get().fetch(CacheTypes.PIPELINE_STATE, Name.get_hash(name));
        if (!pipeline_state) {
            pipeline_state = new PipelineState();
            pipeline_state.init_render_pipeline(context, name, config);
            ResourceCache.get().store(CacheTypes.PIPELINE_STATE, Name.get_hash(name), pipeline_state);
        }
        return pipeline_state;
    }

    static create_compute(context, name, config) {
        let pipeline_state = ResourceCache.get().fetch(CacheTypes.PIPELINE_STATE, Name.get_hash(name));
        if (!pipeline_state) {
            pipeline_state = new PipelineState();
            pipeline_state.init_compute_pipeline(context, name, config);
            ResourceCache.get().store(CacheTypes.PIPELINE_STATE, Name.get_hash(name), pipeline_state);
        }
        return pipeline_state;
    }
}

export default PipelineState;