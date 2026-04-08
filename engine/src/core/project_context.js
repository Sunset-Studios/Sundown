import engine_cvar_config from "../../config/cvars.js";
import { CVarSystem } from "./cvar_system.js";
import { Shader } from "../renderer/shader.js";
import { get_project_shader_root_asset_path } from "../renderer/cooked_asset_config.js";

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

    const project_shader_root_asset_path = get_project_shader_root_asset_path(this.active_project.root);
    if (project_shader_root_asset_path) {
      Shader.register_optional_shader_path(project_shader_root_asset_path);
    }

    return this.active_project;
  }

  static get_active() {
    return this.active_project;
  }
}
