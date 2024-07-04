import { EntityManager } from '@/core/ecs/entity';

const application_state = {
    is_running: false,
    current_view: null,
    entity_manager: new EntityManager()
};

export default application_state;