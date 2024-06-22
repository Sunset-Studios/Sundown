import Renderer from '@/renderer/renderer.js';
import { SimulationCore } from '@/core/simulation_core.js';
import { Scene } from '@/core/layers/scene.js';
import application_state from '@/core/application_state.js';

function init() {
    application_state.is_running = true;

    const canvas = document.getElementById('gpu-canvas');
    Renderer.get().setup(canvas);

    const scene = new Scene("TestScene");
    SimulationCore.get().register_simulation_layer(scene);
}

function run() {
    if (application_state.is_running) {
        {
            console.profile('simulation_core_update');

            Renderer.get().render();
            SimulationCore.get().update();

            console.profileEnd('simulation_core_update');
        }

        window.requestAnimationFrame(run);
    } else {
        cleanup();
    }
}

function cleanup() {
    Renderer.get().cleanup();
    SimulationCore.get().cleanup();
}

init();
run();