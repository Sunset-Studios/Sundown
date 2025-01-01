import { EntityQuery } from "./query.js";

export class EntityManager {
    next_entity_id = 0;
    entity_fragments = new Map();
    fragment_types = new Set();
    deleted_entities = new Set();
    entities = [];
    queries = [];

    constructor() {
        if (EntityManager.instance) {
            return EntityManager.instance;
        }
        EntityManager.instance = this;
    }

    static get() {
        if (!EntityManager.instance) {
            return new EntityManager()
        }
        return EntityManager.instance;
    }

    create_entity(refresh_entity_data = true) {
        let entity;
        if (this.deleted_entities.size > 0) {
            entity = this.deleted_entities.values().next().value;
            this.deleted_entities.delete(entity);
        } else {
            entity = this.next_entity_id++;
            // Resize all fragment data arrays to fit the new entity
            if (refresh_entity_data) {
                for (const fragment_type of this.fragment_types) {
                    fragment_type.resize?.(entity);
                }
            }
        }
        this.entities.push(entity);
        this.entity_fragments.set(entity, new Set());
        if (refresh_entity_data) {
            this.update_queries({ entity });
        }
        return entity;
    }

    delete_entity(entity, refresh_entity_data = true) {
        if (!this.entity_fragments.has(entity)) {
            return;
        }
        for (const FragmentType of this.entity_fragments.get(entity)) {
            FragmentType.remove_entity?.(entity);
        }
        this.entity_fragments.delete(entity);
        this.entities.splice(this.entities.indexOf(entity), 1);
        this.deleted_entities.add(entity);
        if (refresh_entity_data) {
            this.update_queries({ entity });
        }
    }

    duplicate_entity(entity, refresh_entity_data = true) {
        const new_entity = this.create_entity(refresh_entity_data);
        for (const FragmentType of this.entity_fragments.get(entity)) {
            if (FragmentType.data) {
                const data = FragmentType.duplicate_entity_data?.(entity);
                const new_frag_view = this.add_fragment(new_entity, FragmentType, refresh_entity_data);
                for (const [key, value] of Object.entries(data)) {
                    if (value !== null) {
                        if (Array.isArray(value)) {
                            new_frag_view[key] = [...value];
                        } else if (typeof value === 'object') {
                            for (const [sub_key, sub_value] of Object.entries(value)) {
                                new_frag_view[key][sub_key] = sub_value;
                            }
                        } else {
                            new_frag_view[key] = value;
                        }
                    } else {
                        new_frag_view[key] = value;
                    }
                }
            } else {
                this.add_tag(new_entity, FragmentType, refresh_entity_data);
            }
        }
        return new_entity;
    }

    add_fragment(entity, FragmentType, refresh_entity_data = true) {
        if (!this.fragment_types.has(FragmentType)) {
            FragmentType.initialize();
            this.fragment_types.add(FragmentType);
        }
        const fragment_view = FragmentType.add_entity(entity);
        this.entity_fragments.get(entity).add(FragmentType);
        if (refresh_entity_data) {
            this.update_queries({ entity });
        }
        return fragment_view;
    }

    remove_fragment(entity, FragmentType, refresh_entity_data = true) {
        if (!this.entity_fragments.has(entity) || !this.entity_fragments.get(entity).has(FragmentType)) {
            return;
        }
        FragmentType.remove_entity(entity);
        this.entity_fragments.get(entity).delete(FragmentType);
        if (refresh_entity_data) {
            this.update_queries({ entity });
        }
    }

    add_tag(entity, Tag, refresh_entity_data = true) {
        if (!this.fragment_types.has(Tag)) {
            this.fragment_types.add(Tag);
        }
        this.entity_fragments.get(entity).add(Tag);
        if (refresh_entity_data) {
            this.update_queries({ entity });
        }
    }

    remove_tag(entity, Tag, refresh_entity_data = true) {
        if (!this.entity_fragments.has(entity) || !this.entity_fragments.get(entity).has(Tag)) {
            return;
        }
        this.entity_fragments.get(entity).delete(Tag);
        if (refresh_entity_data) {
            this.update_queries({ entity });
        }
    }

    get_fragment(entity, FragmentType) {
        if (!this.entity_fragments.has(entity) || !this.entity_fragments.get(entity).has(FragmentType)) {
            return null;
        }
        return FragmentType.get_entity_data(entity);
    }

    has_fragment(entity, FragmentType) {
        return this.entity_fragments.has(entity) && this.entity_fragments.get(entity).has(FragmentType);
    }

    get_fragment_array(FragmentType) {
        return FragmentType.data;
    }

    get_entity_count() {
        return this.entities.length;
    }

    get_entities() {
        return this.entities;
    }

    create_query({ fragment_requirements }) {
        const query = new EntityQuery(this, fragment_requirements);
        this.queries.push(query);
        return query;
    }

    update_queries(params = {}) {
        for (let i = 0; i < this.queries.length; i++) {
            this.queries[i].update_matching_entities(params);
        }
    }

    process_query_changes() {
        for (let i = 0; i < this.queries.length; i++) {
            this.queries[i].process_entity_changes();
        }
    }

    get_entity_image_buffer() {
        return Buffer.create(context, {
            name: "entity_image_buffer",
            raw_data: this.get_entity_count() * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }
}

// Example usage
// const entity_manager = new EntityManager();

// const entity1 = entity_manager.create_entity();
// entity_manager.add_fragment(entity1, PositionFragment);
// entity_manager.add_fragment(entity1, VelocityFragment);

// const entity2 = entity_manager.create_entity();
// entity_manager.add_fragment(entity2, PositionFragment);

// const query = entity_manager.create_query({ fragment_requirements: [PositionFragment, VelocityFragment] });

// // System example: update positions
// function update_positions(entity_manager, delta_time) {
//     const positions = entity_manager.get_fragment_array(PositionFragment);
//     const velocities = entity_manager.get_fragment_array(VelocityFragment);

//     for (const entity of query) {
//         positions.x[entity] += velocities.vx[entity] * delta_time;
//         positions.y[entity] += velocities.vy[entity] * delta_time;
//     }
// }

// // Usage
// update_positions(entity_manager, 0.16);
