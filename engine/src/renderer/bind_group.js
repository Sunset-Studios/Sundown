import Name from "@/utility/names.js";
import { ResourceCache, CacheTypes } from "@/renderer/resource_cache.js";

export const BindlessGroupIndex = {
    Image: 0,
    StorageImage: 1
}; 

export const BindGroupType = {
    Global: 0,
    Pass: 1
};

export class BindingTableEntry {
    constructor(count) {
        this.total_bindings_count = count;
        this.free_indices = Array.from({length: count}, (_, i) => i);
        this.bound_indices = [];
    }
}

export class GroupBindingTable {
    constructor() {
        this.binding_table = new Map();
    }

    has_binding_slot(slot) {
        return this.binding_table.has(slot);
    }

    add_binding_slot(slot, count) {
        if (!this.has_binding_slot(slot)) {
            this.binding_table.set(slot, new BindingTableEntry(count));
        }
    }

    get_new(slot) {
        if (!this.has_binding_slot(slot)) {
            throw new Error(`Binding slot ${slot} does not exist`);
        }

        const entry = this.binding_table.get(slot);
        if (entry.free_indices.length === 0) {
            throw new Error(`No free indices available for binding slot ${slot}`);
        }

        const index = entry.free_indices.pop();
        entry.bound_indices.push(index);
        return { slot, index };
    }

    free(handle) {
        const { slot, index } = handle;
        if (!this.has_binding_slot(slot)) {
            throw new Error(`Binding slot ${slot} does not exist`);
        }

        const entry = this.binding_table.get(slot);
        const bound_index = entry.bound_indices.indexOf(index);
        if (bound_index === -1) {
            throw new Error(`Index ${index} is not bound for slot ${slot}`);
        }

        entry.bound_indices.splice(bound_index, 1);
        entry.free_indices.push(index);
    }

    reset(slot) {
        if (!this.has_binding_slot(slot)) {
            throw new Error(`Binding slot ${slot} does not exist`);
        }

        const entry = this.binding_table.get(slot);
        entry.free_indices = Array.from({length: entry.total_bindings_count}, (_, i) => i);
        entry.bound_indices = [];
    }
}

export class BindGroup {
    bind_group = null;
    index = 0;
    binding_table = null;

    init(context, name, pipeline, index, bindings) {
        this.index = index;
        this.bind_group = context.device.createBindGroup({
            label: name,
            layout: pipeline.pipeline.getBindGroupLayout(index),
            entries: bindings,
        });
        this.binding_table = new GroupBindingTable();
    }

    bind(render_pass) {
        render_pass.pass.setBindGroup(this.index, this.bind_group);
    }

    static create(context, name, pipeline, index, bindings) {
        let bind_group = ResourceCache.get().fetch(CacheTypes.BIND_GROUP, Name.from(name));
        if (!bind_group) {
            bind_group = new BindGroup();
            bind_group.init(context, name, pipeline, index, bindings);
            ResourceCache.get().store(CacheTypes.BIND_GROUP, Name.from(name), bind_group);
        }
        return bind_group;
    }
}