import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";
import Name from "@/utility/names.js";

class MeshTask {
    constructor(mesh_id) {
        this.mesh_id = mesh_id;
        this.instance_count = 1;
    }
}

export class MeshTaskQueue {
    constructor() {
        if (MeshTaskQueue.instance) {
            return MeshTaskQueue.instance;
        }
        this.tasks = [];
        MeshTaskQueue.instance = this;
    }

    static get() {
        if (!MeshTaskQueue.instance) {
            MeshTaskQueue.instance = new MeshTaskQueue()
        }
        return MeshTaskQueue.instance;
    }

    new_task(mesh) {
        const mesh_id = Name.from(mesh.name);
        this.tasks.forEach(task => {
            if (task.mesh_id === mesh_id) {
                task.instance_count++;
                return task;
            }
        });
        const task = new MeshTask(mesh_id);
        this.tasks.push(task);
        return task;
    }

    remove_single_task(mesh_id) {
        this.tasks = this.tasks.forEach(task => {
            if (task.mesh_id === mesh_id) {
                task.instance_count--;
            }
        });
    }

    remove_all_tasks(mesh_id) {
        this.tasks = this.tasks.filter(task => task.mesh_id !== mesh_id);
    }

    sort() {
        this.tasks.sort((a, b) => a.mesh_id < b.mesh_id);
    }
    
    submit_draws(render_pass) {
        this.tasks.forEach(task => {
            const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);   
            render_pass.pass.draw(mesh.vertices.length, task.instance_count, mesh.vertex_buffer_offset);
        });
        this.tasks.length = 0;
    }

    submit_indexed_draws(render_pass) {
        this.tasks.forEach(task => {
            const mesh = ResourceCache.get().fetch(CacheTypes.MESH, task.mesh_id);   
            render_pass.pass.setIndexBuffer(mesh.index_buffer.buffer, mesh.index_buffer.config.element_type);
            render_pass.pass.drawIndexed(mesh.indices.length, task.instance_count, 0, mesh.vertex_buffer_offset);
        });
        this.tasks.length = 0;
    }
}