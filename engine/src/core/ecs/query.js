export class EntityQuery {
    constructor(entity_manager, fragment_requirements) {
        this.entity_manager = entity_manager;
        this.fragment_requirements = fragment_requirements;
        this.matching_entities = new Set();
        this.update_matching_entities();
    }

    update_matching_entities() {
        this.matching_entities.clear();
        for (let entity = 0; entity < this.entity_manager.get_entity_count(); entity++) {
            if (this.fragment_requirements.every(FragmentType => 
                this.entity_manager.entity_fragments.get(entity)?.has(FragmentType))) {
                this.matching_entities.add(entity);
            }
        }
    }

    get_entity_count() {
        return this.matching_entities.size;
    }

    *[Symbol.iterator]() {
        for (const entity of this.matching_entities) {
            yield entity;
        }
    }
}