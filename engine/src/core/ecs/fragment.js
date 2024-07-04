export class Fragment {
    static data = null;
    static size = 0;
    static entity_set = new Set();

    static initialize() { }

    static resize() { }

    static to_gpu_data() { }

    static add_entity(entity) {
        this.entity_set.add(entity);
        if (entity >= this.size) {
            this.size = entity + 1;
            this.resize();
        }
    }

    static remove_entity(entity) {
        this.entity_set.delete(entity);
        if (entity === this.size - 1) {
            this.size = Math.max(...this.entity_set) + 1;
            this.resize();
        }
    }

    static update_entity_data(entity, data) {
        for (const [key, value] of Object.entries(data)) {
            if (this.data[key] !== undefined) {
                this.data[key][entity] = value;
            } else {
                throw new Error(`Invalid property ${key} for fragment ${this.constructor.name}`);
            }
        }
    }

    static get_entity_data(entity) {
        const fragment_data = {};
        for (const key in this.data) {
            fragment_data[key] = this.data[key][entity];
        }
        return fragment_data;
    }
}