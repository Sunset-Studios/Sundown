import { Renderer } from "./renderer.js";
import { Name } from "../utility/names.js";
import { ResourceCache } from "./resource_cache.js";
import { CacheTypes } from "./renderer_types.js";

export class PipelineState {
    pipeline = null;
    layout = null;
    
    init_render_pipeline(name, config) {
        const renderer = Renderer.get();

        if (config.bind_layouts && config.bind_layouts.length) {
            this.layout = renderer.device.createPipelineLayout({
                label: Name.string(name),
                bindGroupLayouts: config.bind_layouts,
            });
        }
        this.pipeline = renderer.device.createRenderPipeline({
            label: Name.string(name),
            layout: this.layout ?? 'auto',
            ...config
        });
    }

    init_compute_pipeline(name, config) {
        const renderer = Renderer.get();

        if (config.bind_layouts && config.bind_layouts.length) {
            this.layout = renderer.device.createPipelineLayout({
                label: Name.string(name),
                bindGroupLayouts: config.bind_layouts,
            });
        }
        this.pipeline = renderer.device.createComputePipeline({
            label: Name.string(name),
            layout: this.layout ?? 'auto',
            ...config
        });
    }

    static create_render(name, config) {
        let name_hash = Name.from(name);
        let pipeline_state = ResourceCache.get().fetch(CacheTypes.PIPELINE_STATE, name_hash);
        if (!pipeline_state) {
            pipeline_state = new PipelineState();
            pipeline_state.init_render_pipeline(name_hash, config);
            ResourceCache.get().store(CacheTypes.PIPELINE_STATE, name_hash, pipeline_state);
        }
        return pipeline_state;
    }

    static create_compute(name, config) {
        let name_hash = Name.from(name);
        let pipeline_state = ResourceCache.get().fetch(CacheTypes.PIPELINE_STATE, name_hash);
        if (!pipeline_state) {
            pipeline_state = new PipelineState();
            pipeline_state.init_compute_pipeline(name_hash, config);
            ResourceCache.get().store(CacheTypes.PIPELINE_STATE, name_hash, pipeline_state);
        }
        return pipeline_state;
    }
}