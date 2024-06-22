export const CacheTypes = Object.freeze({
    SHADER: 0,
    PIPELINE_STATE: 1,
    RENDER_PASS: 2,
    BUFFER: 3,
    TEXTURE: 4,
    SAMPLER: 5,
    DESCRIPTOR_SET_LAYOUT: 6,
    DESCRIPTOR_SET: 7,
});

export class ResourceCache {
    constructor() {
        if (ResourceCache.instance) {
            return ResourceCache.instance;
        }
        ResourceCache.instance = this;

        this.cache = new Map();
        this.cache.set(CacheTypes.SHADER, new Map());
        this.cache.set(CacheTypes.PIPELINE_STATE, new Map());
        this.cache.set(CacheTypes.RENDER_PASS, new Map());
        this.cache.set(CacheTypes.BUFFER, new Map());
        this.cache.set(CacheTypes.TEXTURE, new Map());
        this.cache.set(CacheTypes.SAMPLER, new Map());
        this.cache.set(CacheTypes.DESCRIPTOR_SET_LAYOUT, new Map());
        this.cache.set(CacheTypes.DESCRIPTOR_SET, new Map());
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
}