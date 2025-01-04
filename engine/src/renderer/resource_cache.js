import { CacheTypes } from "./renderer_types.js";

export class ResourceCache {
    constructor() {
        if (ResourceCache.instance) {
            return ResourceCache.instance;
        }
        ResourceCache.instance = this;

        this.cache = new Map();
        this.cache.set(CacheTypes.SHADER, new Map());
        this.cache.set(CacheTypes.PIPELINE_STATE, new Map());
        this.cache.set(CacheTypes.PASS, new Map());
        this.cache.set(CacheTypes.BIND_GROUP, new Map());
        this.cache.set(CacheTypes.BIND_GROUP_LAYOUT, new Map());
        this.cache.set(CacheTypes.BUFFER, new Map());
        this.cache.set(CacheTypes.IMAGE, new Map());
        this.cache.set(CacheTypes.SAMPLER, new Map());
        this.cache.set(CacheTypes.MESH, new Map());
        this.cache.set(CacheTypes.MATERIAL, new Map());
    }

    static get() {
        if (!ResourceCache.instance) {
            ResourceCache.instance = new ResourceCache()
        }
        return ResourceCache.instance;
    }

    fetch(type, key) {
        return this.cache.get(type).get(key);
    }

    store(type, key, value) {
        this.cache.get(type).set(key, value);
    }

    remove(type, key) {
        this.cache.get(type).delete(key);
    }

    size(type) {
        return this.cache.get(type).size;
    }

    flush(type) {
        const keys = this.cache.get(type).keys();   
        for (const key of keys) {
            const resource = this.cache.get(type).get(key);
            resource.destroy?.();
            this.cache.get(type).delete(key);
        }
    }
}