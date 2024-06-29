import { ResourceCache, CacheTypes } from '@/renderer/resource_cache.js';

export class Shader {
    static shader_paths = ['/shaders'];

    module = null;
    code = null;
    file_path = '';

    static register_shader_path(path) {
        Shader.shader_paths.push(path);
    }

    initialize(context, file_path) {
        let asset = null;
        for (const path of Shader.shader_paths) {
            try {
                const response = new XMLHttpRequest();
                response.open('GET', `${path}/${file_path}`, false);
                response.send(null);
                if (response.status === 200) {
                    asset = response.responseText;
                    break;
                }
            } catch (error) {
                console.warn(`Failed to load shader from ${path}/${file_path}:`, error);
            }
        }

        if (!asset) {
            console.error(`WebGPU shader error: could not find shader at ${file_path}`);
            return;
        }

        try {
            this.code = asset;
            this.module = context.device.createShaderModule({
                label: file_path,
                code: asset
            });
            this.file_path = file_path;
        } catch (error) {
            console.error(`WebGPU shader error: could not create shader module at ${file_path}`, error);
        }
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