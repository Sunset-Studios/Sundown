import { RenderPassFlags } from "../renderer_types.js";

/**
 * Base class for one complete GI module.
 *
 * Modules own their resources and render-graph passes. Semantic names stay
 * independent from physical buffer names so a module can change layouts and
 * shaders without changing its consumers.
 */
export class GIModule {
  constructor({ name, representation, shader_setups = {}, stages = {} }) {
    if (!name || !representation) {
      throw new Error("GI modules require a name and representation");
    }

    this.name = name;
    this.representation = representation;
    this.shader_setups = { ...shader_setups };
    this.resources = new Map();
    this.frame_context = null;
    this.stages = {
      trace: true,
      shading: true,
      accumulation: true,
      ...stages,
    };
    this._validate_stages();
  }

  begin_frame(frame_context) {
    this.resources.clear();
    this.frame_context = frame_context;
  }

  setup(_render_graph, _frame_context, _branch) { }

  record(_render_graph, _frame_context, _branch) { }

  record_debug(_render_graph, _debug_context, _branch) { return null; }

  create_buffer(render_graph, semantic, config) {
    const handle = render_graph.create_buffer(config);
    this.resources.set(semantic, handle);
    return handle;
  }

  create_image(render_graph, semantic, config) {
    const handle = render_graph.create_image(config);
    this.resources.set(semantic, handle);
    return handle;
  }

  import_resource(semantic, handle) {
    this.resources.set(semantic, handle);
    return handle;
  }

  get_resource(semantic) {
    return this.resources.get(semantic) ?? null;
  }

  get_shader_setup(semantic, fallback = null) {
    return this.shader_setups[semantic] ?? fallback;
  }

  add_compute_pass(render_graph, semantic, name, parameters, callback, fallback_shader = null) {
    const shader_setup = this.get_shader_setup(semantic, fallback_shader);
    if (!shader_setup) {
      throw new Error(`${this.name} does not provide a shader for '${semantic}'`);
    }

    return render_graph.add_pass(
      name,
      RenderPassFlags.Compute,
      { ...parameters, shader_setup },
      callback
    );
  }

  add_graph_local_pass(render_graph, name, callback) {
    return render_graph.add_pass(name, RenderPassFlags.GraphLocal, {}, callback);
  }

  _validate_stages() {
    const known_stages = new Set(Object.values(GIPipelineStage));
    for (const stage of Object.keys(this.stages)) {
      if (!known_stages.has(stage)) {
        throw new Error(`Unknown GI pipeline stage '${stage}'`);
      }
    }
    for (const stage of known_stages) {
      if (typeof this.stages[stage] !== "boolean") {
        throw new Error(`${this.name} stage '${stage}' must be enabled or disabled`);
      }
    }
  }

  is_stage_enabled(stage) {
    if (!Object.values(GIPipelineStage).includes(stage)) {
      throw new Error(`Unknown GI pipeline stage '${stage}'`);
    }
    return this.stages[stage];
  }

  set_stage_enabled(stage, enabled) {
    if (!Object.values(GIPipelineStage).includes(stage)) {
      throw new Error(`Unknown GI pipeline stage '${stage}'`);
    }
    if (typeof enabled !== "boolean") {
      throw new Error(`${this.name} stage '${stage}' must be enabled or disabled`);
    }
    this.stages[stage] = enabled;
    return this;
  }

  set_stages(stages) {
    for (const [stage, enabled] of Object.entries(stages)) {
      this.set_stage_enabled(stage, enabled);
    }
    return this;
  }
}

export const GIPipelineStage = Object.freeze({
  Trace: "trace",
  Shading: "shading",
  Accumulation: "accumulation",
});

/**
 * A branch contains one complete GI module. A pipeline may contain multiple
 * branches (PTGI uses a surface-cache and a per-pixel radiance-cache module).
 */
export class GIPipelineBranch {
  constructor({ name, module, dependencies = {} }) {
    this.name = name;
    this.module = module;
    this.dependency_names = { ...dependencies };
    this.dependencies = new Map();
    this.validate();
  }

  resolve_dependencies(branches) {
    this.dependencies.clear();
    for (const [semantic, branch_name] of Object.entries(this.dependency_names)) {
      const dependency = branches.get(branch_name);
      if (!dependency) {
        throw new Error(`GI branch '${this.name}' requires unknown branch '${branch_name}'`);
      }
      this.dependencies.set(semantic, dependency);
    }
  }

  get_dependency(semantic) {
    const dependency = this.dependencies.get(semantic);
    if (!dependency) {
      throw new Error(`GI branch '${this.name}' does not provide dependency '${semantic}'`);
    }
    return dependency;
  }

  validate() {
    if (!(this.module instanceof GIModule)) {
      throw new Error(`GI branch '${this.name}' requires a GI module`);
    }
  }

  begin_frame(frame_context) {
    this.module.begin_frame(frame_context, this);
  }
}

export class GIPipelineComposition {
  branches = null;
  branch_map = null;
  
  constructor(branches) {
    this.branches = branches.map((branch) =>
      branch instanceof GIPipelineBranch ? branch : new GIPipelineBranch(branch)
    );

    this.branch_map = new Map();
    for (let i = 0; i < this.branches.length; ++i) {
      this.branch_map.set(this.branches[i].name, this.branches[i]);
    }

    for (let i = 0; i < this.branches.length; ++i) {
      this.branches[i].resolve_dependencies(this.branch_map);
    }
  }

  begin_frame(frame_context) {
    // A module may be shared by multiple named branches. Begin it only once so
    // its semantic resource registry remains stable for the frame.
    const branches_started = new Set();
    for (let i = 0; i < this.branches.length; ++i) {
      const branch = this.branches[i];
      if (branches_started.has(branch.module)) {
        continue;
      }

      branch.begin_frame(frame_context);
      branches_started.add(branch.module);
    }
  }

  add_passes(render_graph, frame_context = {}) {
    frame_context.render_graph = render_graph;

    this.begin_frame(frame_context);

    // Allocate every module first so dependent branches can consume one
    // another's stable semantic resources when passes are recorded.
    const setup_modules = new Set();
    for (let i = 0; i < this.branches.length; ++i) {
      const branch = this.branches[i];
      if (!setup_modules.has(branch.module)) {
        setup_modules.add(branch.module);
        branch.module.setup(render_graph, frame_context, branch);
      }
    }

    // Branch order is significant and explicit. PTGI records its surface-cache
    // branch before the per-pixel branch that samples it.
    const recorded_modules = new Set();
    for (let i = 0; i < this.branches.length; ++i) {
      const branch = this.branches[i];
      if (!recorded_modules.has(branch.module)) {
        branch.module.record(render_graph, frame_context, branch);
        recorded_modules.add(branch.module);
      } 
    }

    return frame_context;
  }

  add_debug_passes(render_graph, debug_context = {}) {
    debug_context.render_graph = render_graph;
    const recorded_modules = new Set();

    for (let i = 0; i < this.branches.length; ++i) {
      const branch = this.branches[i];
      if (!recorded_modules.has(branch.module)) {
        recorded_modules.add(branch.module);
        const output = branch.module.record_debug(render_graph, debug_context, branch);
        if (output) return output;
      } 
    }

    return null;
  }

  get_branch(name) {
    const branch = this.branch_map.get(name);
    return !!branch ? branch : null;
  }

  get_module(name) {
    const branch = this.get_branch(name);
    return branch?.module ?? null;
  }
}
