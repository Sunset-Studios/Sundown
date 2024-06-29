import { glTFLoader } from '@/utility/gltf_loader.js';
import { ResourceCache, CacheTypes } from '@/renderer/resource_cache.js';
import Name from '@/utility/names.js';

export class Mesh {
    constructor() {
        this.vertices = [];
        this.indices = [];
    }

    static async from_gltf(gltf) {
        let mesh = ResourceCache.get().fetch(CacheTypes.MESH, Name.from(gltf));
        if (mesh) {
            return mesh;
        }

        return new Promise((resolve) => {
            const loader = new glTFLoader();
            loader.load(gltf, (gltf_obj) => {
                mesh = new Mesh();
                mesh.vertices = gltf_obj.vertices;
                mesh.indices = gltf_obj.indices;
                ResourceCache.get().store(CacheTypes.MESH, Name.from(gltf), mesh);
                resolve(mesh);
            });
        });
    }
}