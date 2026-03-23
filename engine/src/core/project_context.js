import engine_cvar_config from "../../config/cvars.js";
import { CVarSystem } from "./cvar_system.js";

const default_project = Object.freeze({
  cvar_config: null,
  name: "default",
  root: "",
});

export class ProjectContext {
  static active_project = default_project;

  static configure(project = null) {
    const normalized_project = {
      ...default_project,
      ...(project || {}),
    };

    if (!normalized_project.cvar_config && normalized_project.cvars) {
      normalized_project.cvar_config = normalized_project.cvars;
    }

    this.active_project = normalized_project;

    CVarSystem.reset_all({ silent: true });
    CVarSystem.apply_config(engine_cvar_config, {
      context: { project: normalized_project },
      source: "engine_defaults",
    });

    if (normalized_project.cvar_config) {
      CVarSystem.apply_config(normalized_project.cvar_config, {
        context: { project: normalized_project },
        source: `project:${normalized_project.name}`,
      });
    }

    return this.active_project;
  }

  static get_active() {
    return this.active_project;
  }
}
