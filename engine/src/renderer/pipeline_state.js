import { Renderer } from "./renderer.js";
import { Name } from "../utility/names.js";
import { ResourceCache } from "./resource_cache.js";
import { CacheTypes } from "./renderer_types.js";
import { error } from "../utility/logging.js";

export class PipelineState {
    pipeline = null;
    layout = null;
    bind_group_layouts = [];
    ready = false;
    pending = false;
    compile_promise = null;
    compile_error = null;

    init_layout(name, bind_layouts = []) {
        const renderer = Renderer.get();

        this.bind_group_layouts = [...bind_layouts];
        this.layout = null;

        if (bind_layouts && bind_layouts.length) {
            this.layout = renderer.device.createPipelineLayout({
                label: Name.string(name),
                bindGroupLayouts: bind_layouts,
            });
        }
    }

    get_bind_group_layout(index) {
        if (index < this.bind_group_layouts.length) {
            return this.bind_group_layouts[index];
        }

        if (this.pipeline) {
            return this.pipeline.getBindGroupLayout(index);
        }

        return null;
    }
    
    init_render_pipeline(name, config) {
        const renderer = Renderer.get();
        const { bind_layouts = [], force, defer_creation = true, ...pipeline_config } = config;

        this.init_layout(name, bind_layouts);
        this.ready = false;
        this.pending = true;
        this.compile_error = null;

        const descriptor = {
            label: Name.string(name),
            layout: this.layout ?? 'auto',
            ...pipeline_config
        };

        const finish = (pipeline) => {
            this.pipeline = pipeline;
            this.ready = true;
            this.pending = false;
            this.compile_error = null;
            return this;
        };

        const fail = (err) => {
            this.pipeline = null;
            this.ready = false;
            this.pending = false;
            this.compile_error = err;
            error(`Failed to create render pipeline '${Name.string(name)}'`, err);
            return this;
        };

        if (defer_creation && renderer.device.createRenderPipelineAsync) {
            this.compile_promise = renderer.device.createRenderPipelineAsync(descriptor)
                .then(finish)
                .catch(fail);
        } else {
            try {
                finish(renderer.device.createRenderPipeline(descriptor));
                this.compile_promise = Promise.resolve(this);
            } catch (err) {
                this.compile_promise = Promise.resolve(fail(err));
            }
        }
    }

    init_compute_pipeline(name, config) {
        const renderer = Renderer.get();
        const { bind_layouts = [], force, defer_creation = true, ...pipeline_config } = config;

        this.init_layout(name, bind_layouts);
        this.ready = false;
        this.pending = true;
        this.compile_error = null;

        const descriptor = {
            label: Name.string(name),
            layout: this.layout ?? 'auto',
            ...pipeline_config
        };

        const finish = (pipeline) => {
            this.pipeline = pipeline;
            this.ready = true;
            this.pending = false;
            this.compile_error = null;
            return this;
        };

        const fail = (err) => {
            this.pipeline = null;
            this.ready = false;
            this.pending = false;
            this.compile_error = err;
            error(`Failed to create compute pipeline '${Name.string(name)}'`, err);
            return this;
        };

        if (defer_creation && renderer.device.createComputePipelineAsync) {
            this.compile_promise = renderer.device.createComputePipelineAsync(descriptor)
                .then(finish)
                .catch(fail);
        } else {
            try {
                finish(renderer.device.createComputePipeline(descriptor));
                this.compile_promise = Promise.resolve(this);
            } catch (err) {
                this.compile_promise = Promise.resolve(fail(err));
            }
        }
    }

    is_ready() {
        return this.ready && this.pipeline !== null;
    }

    when_ready() {
        return this.compile_promise ?? Promise.resolve(this);
    }

    static create_render(name, config) {
        let name_hash = Name.from(name);
        let pipeline_state = ResourceCache.get().fetch(CacheTypes.PIPELINE_STATE, name_hash);

        if (pipeline_state && config.force) {
            pipeline_state = null;
        }

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

        if (pipeline_state && config.force) {
            pipeline_state = null;
        }

        if (!pipeline_state) {
            pipeline_state = new PipelineState();
            pipeline_state.init_compute_pipeline(name_hash, config);
            ResourceCache.get().store(CacheTypes.PIPELINE_STATE, name_hash, pipeline_state);
        }

        return pipeline_state;
    }
}