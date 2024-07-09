import { EntityQuery } from "@/core/ecs/query.js";

export class EntityManager {
    next_entity_id = 0;
    entity_fragments = new Map();
    fragment_types = new Set();
    queries = new Set();
    deleted_entities = new Set();

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

    create_entity() {
        let entity;
        if (this.deleted_entities.size > 0) {
            entity = this.deleted_entities.values().next().value;
            this.deleted_entities.delete(entity);
        } else {
            entity = this.next_entity_id++;
        }
        this.entity_fragments.set(entity, new Set());
        this.update_queries();
        return entity;
    }

    delete_entity(entity) {
        if (!this.entity_fragments.has(entity)) {
            return;
        }
        for (const FragmentType of this.entity_fragments.get(entity)) {
            FragmentType.remove_entity(entity);
        }
        this.entity_fragments.delete(entity);
        this.deleted_entities.add(entity);
        this.update_queries();
    }

    add_fragment(entity, FragmentType, data) {
        if (!this.fragment_types.has(FragmentType)) {
            FragmentType.initialize();
            this.fragment_types.add(FragmentType);
        }
        FragmentType.add_entity(entity, data);
        this.entity_fragments.get(entity).add(FragmentType);
        this.update_queries();
    }

    remove_fragment(entity, FragmentType) {
        if (!this.entity_fragments.has(entity) || !this.entity_fragments.get(entity).has(FragmentType)) {
            return;
        }
        FragmentType.remove_entity(entity);
        this.entity_fragments.get(entity).delete(FragmentType);
        this.update_queries();
    }

    update_fragment(entity, FragmentType, data) {
        if (!this.entity_fragments.has(entity) || !this.entity_fragments.get(entity).has(FragmentType)) {
            throw new Error(`Entity ${entity} does not have fragment ${FragmentType.constructor.name}`);
        }
        FragmentType.update_entity_data(entity, data);
    }

    get_fragment(entity, FragmentType) {
        if (!this.entity_fragments.has(entity) || !this.entity_fragments.get(entity).has(FragmentType)) {
            return null;
        }
        return FragmentType.get_entity_data(entity);
    }

    get_fragment_array(FragmentType) {
        return FragmentType.data;
    }

    get_entity_count() {
        return this.next_entity_id - this.deleted_entities.size;
    }

    create_query({ fragment_requirements }) {
        const query = new EntityQuery(this, fragment_requirements);
        this.queries.add(query);
        return query;
    }

    update_queries() {
        for (const query of this.queries) {
            query.update_matching_entities();
        }
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
