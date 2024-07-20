import { Name } from "../utility/names.js";
import { ResourceCache, CacheTypes } from "./resource_cache.js";

export const BindlessGroupIndex = {
    Image: 0,
    StorageImage: 1
}; 

export const BindGroupType = {
    Global: 0,
    Pass: 1,
    Material: 2,
    Num: 3
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
    index = 0;
    name = ''
    bind_group = null;
    layout = null;
    binding_table = null;

    init(context, name, pipeline, index, bindings) {
        this.name = name;
        this.index = index;
        this.layout = pipeline.pipeline.getBindGroupLayout(index);
        this.bind_group = context.device.createBindGroup({
            label: name,
            layout: this.layout,
            entries: bindings,
        });
        this.binding_table = new GroupBindingTable();
    }

    init_with_layout(context, name, layout, index, bindings) {
        this.name = name;
        this.index = index;
        this.layout = BindGroup.create_layout(context, name, layout);
        this.bind_group = context.device.createBindGroup({
            label: name,
            layout: this.layout,
            entries: bindings,
        });
        this.binding_table = new GroupBindingTable();
    }

    destroy() {
        if (this.bind_group) {
            ResourceCache.get().remove(CacheTypes.BIND_GROUP, Name.from(this.name));

            this.bind_group = null;
            this.layout = null;
            this.binding_table = null;
        }
    }

    bind(render_pass) {
        render_pass.pass.setBindGroup(this.index, this.bind_group);
    }

    static create(context, name, pipeline, index, bindings, force = false) {
        let bind_group = ResourceCache.get().fetch(CacheTypes.BIND_GROUP, Name.from(name));

        if (bind_group && force) {
            bind_group.destroy();
            bind_group = null;
        }

        if (!bind_group) {
            bind_group = new BindGroup();
            bind_group.init(context, name, pipeline, index, bindings);
            ResourceCache.get().store(CacheTypes.BIND_GROUP, Name.from(name), bind_group);
        }

        return bind_group;
    }

    static create_with_layout(context, name, layout, index, bindings, force = false) {
        let bind_group = ResourceCache.get().fetch(CacheTypes.BIND_GROUP, Name.from(name));

        if (bind_group && force) {
            bind_group.destroy()
            bind_group = null;
        }

        if (!bind_group) {
            bind_group = new BindGroup();
            bind_group.init_with_layout(context, name, layout, index, bindings);
            ResourceCache.get().store(CacheTypes.BIND_GROUP, Name.from(name), bind_group);
        }

        return bind_group;
    }

    static create_layout(context, name, bind_layouts, force = false) {
        let layout = ResourceCache.get().fetch(CacheTypes.BIND_GROUP_LAYOUT, Name.from(name));

        if (layout && force) {
            layout.destroy();
            layout = null;
        }

        if (!layout) {
            layout = context.device.createBindGroupLayout({
                label: name,
                entries: bind_layouts,
            });
            ResourceCache.get().store(CacheTypes.BIND_GROUP_LAYOUT, Name.from(name), layout);
        }

        return layout;
    }
}