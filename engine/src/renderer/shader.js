import { ResourceCache, CacheTypes } from './resource_cache.js';

export class Shader {
    static shader_paths = ['engine/shaders'];

    module = null;
    code = null;
    file_path = '';

    static register_shader_path(path) {
        Shader.shader_paths.push(path);
    }

    initialize(context, file_path) {
        let asset = this._load_shader_text(file_path);
        if (!asset) {
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

    _load_shader_text(file_path) {
        let asset = null;
        for (const path of Shader.shader_paths) {
            try {
                const url = new URL(`${path}/${file_path}`, window.location.href);
                const response = new XMLHttpRequest();
                response.open('GET', url.href, false);
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
            return null;
        }
        
        asset = this._parse_shader_includes(asset);

        return asset;
    }

    _parse_shader_includes(code) {
        let include_positions = [];

        let pos = code.indexOf("#include", 0);
        while (pos !== -1) {
            include_positions.push(pos);
            pos = code.indexOf("#include", pos + 1);
        }

        const include_regex = /^#include\s+"(\S+)".*$/m;

        for (let i = include_positions.length - 1; i >= 0; --i) {
            const start = include_positions[i];
            const end = code.indexOf('\n', start);
            const include_line = code.substring(start, end);
            const match = include_line.match(include_regex);
            if (match) {
                const include_contents = this._load_shader_text(match[1]);
                code = code.slice(0, start) + include_contents + code.slice(end);
            }
        }

        return code;
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