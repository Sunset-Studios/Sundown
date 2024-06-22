import { ResourceCache, CacheTypes } from '@/renderer/resource_cache.js';

class Shader {
    static shader_paths = [];

    module = null;
    code = null;
    file_path = '';

    constructor() { }

    static register_shader_path(path) {
        Shader.shader_paths.push(path);
    }

    async initialize(context, file_path) {
        let asset = null;
        for (const path of Shader.shader_paths) {
            asset = await fetch(path + file_path);
            if (asset) {
                break;
            }
        }

        if (!asset) {
            console.error(`WebGPU shader error: could not find shader at ${file_path}`);
            return;
        }

        try {
            this.code = asset.body;
            this.module = await context.device.createShaderModule({
                label: file_path,
                code: asset.body
            });
            this.file_path = file_path;
        } catch (error) {
            console.error(`WebGPU shader error: could not create shader module at ${file_path}`, error);
        }
    }

    destroy() {
        this.module = null;
    }

    static create(context, file_path) {
        let shader = ResourceCache.get().fetch(CacheTypes.SHADER, file_path);
        if (!shader) {
            shader = new Shader();
            shader.initialize(context, file_path);
            ResourceCache.get().store(CacheTypes.SHADER, file_path, shader);
        }
        return shader;
    }
}

export default Shader;