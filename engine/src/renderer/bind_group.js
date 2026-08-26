import { Renderer } from "./renderer.js";
import { Name } from "../utility/names.js";
import { ResourceCache } from "./resource_cache.js";
import { CacheTypes } from "./renderer_types.js";

export class BindGroup {
    index = 0;
    name = ''
    bind_group = null;
    layout = null;

    init(name, pipeline, index, bindings) {
        const renderer = Renderer.get();

        this.name = name;
        this.index = index;
        this.layout = pipeline.get_bind_group_layout(index);
        if (!this.layout) {
            throw new Error(`Bind group layout ${index} is not ready for pipeline '${name}'`);
        }

        this.bind_group = renderer.device.createBindGroup({
            label: name,
            layout: this.layout,
            entries: bindings,
        });
    }

    init_with_layout(
        name,
        layout,
        index,
        bindings,
        force = false,
        layout_name = name,
        force_layout = force
    ) {
        const renderer = Renderer.get();

        this.name = name;
        this.index = index;
        this.layout = BindGroup.create_layout(layout_name, layout, force_layout);
        this.bind_group = renderer.device.createBindGroup({
            label: name,
            layout: this.layout,
            entries: bindings,
        });
    }

    destroy() {
        if (this.bind_group) {
            ResourceCache.get().remove(CacheTypes.BIND_GROUP, Name.from(this.name));

            this.bind_group = null;
            this.layout = null;
        }
    }

    bind(render_pass) {
        render_pass.pass.setBindGroup(this.index, this.bind_group);
    }

    static create(name, pipeline, index, bindings, force = false) {
        let bind_group = ResourceCache.get().fetch(CacheTypes.BIND_GROUP, Name.from(name));

        if (bind_group && force) {
            bind_group.destroy();
            bind_group = null;
        }

        if (!bind_group) {
            bind_group = new BindGroup();
            bind_group.init(name, pipeline, index, bindings);
            ResourceCache.get().store(CacheTypes.BIND_GROUP, Name.from(name), bind_group);
        }

        return bind_group;
    }

    static create_with_layout(
        name,
        layout,
        index,
        bindings,
        force = false,
        layout_name = name,
        force_layout = force
    ) {
        let bind_group = ResourceCache.get().fetch(CacheTypes.BIND_GROUP, Name.from(name));

        if (bind_group && force) {
            bind_group.destroy()
            bind_group = null;
        }

        if (!bind_group) {
            bind_group = new BindGroup();
            bind_group.init_with_layout(
                name,
                layout,
                index,
                bindings,
                force,
                layout_name,
                force_layout
            );
            ResourceCache.get().store(CacheTypes.BIND_GROUP, Name.from(name), bind_group);
        }

        return bind_group;
    }

    static create_layout(name, bind_layouts, force = false) {
        const renderer = Renderer.get();

        let layout = ResourceCache.get().fetch(CacheTypes.BIND_GROUP_LAYOUT, Name.from(name));

        if (layout && force) {
            layout = null;
        }

        if (!layout) {
            layout = renderer.device.createBindGroupLayout({
                label: name,
                entries: bind_layouts,
            });
            ResourceCache.get().store(CacheTypes.BIND_GROUP_LAYOUT, Name.from(name), layout);
        }

        return layout;
    }
}
