export class EntityQuery {
    constructor(entity_manager, fragment_requirements) {
        this.entity_manager = entity_manager;
        this.fragment_requirements = fragment_requirements;
        this.matching_entities = new Uint32Array();
        this.update_matching_entities();
    }

    update_matching_entities() {
        const entity_count = this.entity_manager.get_entity_count();
        const new_matching_entities = new Uint32Array(entity_count);
        let matching_count = 0;

        const seen_entities = new Set();
        for (let entity = 0; entity < entity_count; entity++) {
            if (!seen_entities.has(entity) && 
                this.fragment_requirements.every(fragment_type => 
                    this.entity_manager.entity_fragments.get(entity)?.has(fragment_type))) {
                new_matching_entities[matching_count] = entity;
                matching_count++;
                seen_entities.add(entity);
            }
        }

        this.matching_entities = new_matching_entities.slice(0, matching_count);
    }

    get_entity_count() {
        return this.matching_entities.length;
    }
}