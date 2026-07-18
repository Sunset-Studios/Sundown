import { RenderPassFlags } from "../renderer_types.js";

/**
 * Base class for one replaceable part of a GI pipeline.
 *
 * Components own the resources and render-graph passes for their stage.  The
 * semantic names deliberately stay independent from physical buffer names so
 * a component can replace layouts and shaders without changing its consumer.
 */
export class GIComponent {
  constructor({ name, representation, shader_setups = {} }) {
    if (!name || !representation) {
      throw new Error("GI components require a name and representation");
    }

    this.name = name;
    this.representation = representation;
    this.shader_setups = { ...shader_setups };
    this.resources = new Map();
    this.pass_count = 0;
    this.frame_context = null;
  }

  begin_frame(frame_context) {
    this.resources.clear();
    this.pass_count = 0;
    this.frame_context = frame_context;
  }

  setup(_render_graph, _frame_context, _branch) {}

  record(_render_graph, _frame_context, _branch) {}

  record_post_accumulation(_render_graph, _frame_context, _branch) {}

  record_resolve(_render_graph, _frame_context, _branch) {}

  record_debug(_render_graph, _debug_context, _branch) {
    return null;
  }

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

    this.pass_count++;
    return render_graph.add_pass(
      name,
      RenderPassFlags.Compute,
      { ...parameters, shader_setup },
      callback
    );
  }

  add_graph_local_pass(render_graph, name, callback) {
    this.pass_count++;
    return render_graph.add_pass(name, RenderPassFlags.GraphLocal, {}, callback);
  }
}

export class GITraceHitCache extends GIComponent {
  constructor(config) {
    super(config);
    this.hit_representation = config.hit_representation;
  }
}

export class GIShadingStrategy extends GIComponent {
  constructor(config) {
    super(config);
    this.accepted_hit_representations = new Set(config.accepted_hit_representations ?? []);
    this.radiance_representation = config.radiance_representation;
  }

  accepts(trace_hit_cache) {
    return this.accepted_hit_representations.has(trace_hit_cache.hit_representation);
  }
}

export class GIAccumulator extends GIComponent {
  constructor(config) {
    super(config);
    this.accepted_radiance_representations = new Set(
      config.accepted_radiance_representations ?? []
    );
  }

  accepts(shading_strategy) {
    return this.accepted_radiance_representations.has(shading_strategy.radiance_representation);
  }
}

/**
 * A branch is trace-hit storage -> shading -> accumulation.  A pipeline may
 * contain multiple branches (PTGI uses a surface-cache and a per-pixel branch).
 */
export class GIPipelineBranch {
  constructor({ name, trace_hit_cache, shading_strategy, accumulator, dependencies = {} }) {
    this.name = name;
    this.trace_hit_cache = trace_hit_cache;
    this.shading_strategy = shading_strategy;
    this.accumulator = accumulator;
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
    if (!(this.trace_hit_cache instanceof GITraceHitCache)) {
      throw new Error(`GI branch '${this.name}' requires a trace-hit cache`);
    }
    if (!(this.shading_strategy instanceof GIShadingStrategy)) {
      throw new Error(`GI branch '${this.name}' requires a shading strategy`);
    }
    if (!(this.accumulator instanceof GIAccumulator)) {
      throw new Error(`GI branch '${this.name}' requires an accumulator`);
    }
    if (!this.shading_strategy.accepts(this.trace_hit_cache)) {
      throw new Error(
        `${this.shading_strategy.name} cannot shade ${this.trace_hit_cache.hit_representation}`
      );
    }
    if (!this.accumulator.accepts(this.shading_strategy)) {
      throw new Error(
        `${this.accumulator.name} cannot accumulate ${this.shading_strategy.radiance_representation}`
      );
    }
  }

  begin_frame(frame_context) {
    this.trace_hit_cache.begin_frame(frame_context);
    this.shading_strategy.begin_frame(frame_context);
    this.accumulator.begin_frame(frame_context);
  }
}

export class GIPipelineComposition {
  constructor(branches) {
    this.branches = branches.map((branch) =>
      branch instanceof GIPipelineBranch ? branch : new GIPipelineBranch(branch)
    );
    this.branch_map = new Map(this.branches.map((branch) => [branch.name, branch]));
    for (const branch of this.branches) branch.resolve_dependencies(this.branch_map);
  }

  begin_frame(frame_context) {
    // A cache may fan out into multiple shading/accumulation branches. Begin a
    // shared component only once so its semantic resource registry is shared.
    const begun = new Set();
    for (const branch of this.branches) {
      for (const component of [
        branch.trace_hit_cache,
        branch.shading_strategy,
        branch.accumulator,
      ]) {
        if (begun.has(component)) continue;
        component.begin_frame(frame_context);
        begun.add(component);
      }
    }
  }

  add_passes(render_graph, frame_context = {}) {
    frame_context.render_graph = render_graph;
    this.begin_frame(frame_context);

    // Allocate every branch first. This lets a trace cache consume accumulator
    // storage (and vice versa) without either component knowing who created it.
    const setup_components = new Set();
    for (const branch of this.branches) {
      for (const component of [
        branch.trace_hit_cache,
        branch.shading_strategy,
        branch.accumulator,
      ]) {
        if (setup_components.has(component)) continue;
        component.setup(render_graph, frame_context, branch);
        setup_components.add(component);
      }
    }

    // Branch order is significant and explicit. PTGI records its surface-cache
    // branch before the per-pixel branch that samples it.
    const recorded_components = new Set();
    const post_accumulation_components = new Set();
    const resolved_components = new Set();
    for (const branch of this.branches) {
      for (const component of [
        branch.trace_hit_cache,
        branch.shading_strategy,
        branch.accumulator,
      ]) {
        if (recorded_components.has(component)) continue;
        component.record(render_graph, frame_context, branch);
        recorded_components.add(component);
      }
      if (!post_accumulation_components.has(branch.trace_hit_cache)) {
        branch.trace_hit_cache.record_post_accumulation(render_graph, frame_context, branch);
        post_accumulation_components.add(branch.trace_hit_cache);
      }
      if (!resolved_components.has(branch.accumulator)) {
        branch.accumulator.record_resolve(render_graph, frame_context, branch);
        resolved_components.add(branch.accumulator);
      }
    }

    return frame_context;
  }

  add_debug_passes(render_graph, debug_context = {}) {
    debug_context.render_graph = render_graph;
    const recorded_components = new Set();

    for (const branch of this.branches) {
      for (const component of [
        branch.trace_hit_cache,
        branch.shading_strategy,
        branch.accumulator,
      ]) {
        if (recorded_components.has(component)) continue;
        recorded_components.add(component);
        const output = component.record_debug(render_graph, debug_context, branch);
        if (output) return output;
      }
    }

    return null;
  }

  get_branch(name) {
    const branch = this.branch_map.get(name);
    if (!branch) throw new Error(`Unknown GI pipeline branch '${name}'`);
    return branch;
  }

  get_components(name) {
    const branch = this.get_branch(name);
    return {
      trace_hit_cache: branch.trace_hit_cache,
      shading_strategy: branch.shading_strategy,
      accumulator: branch.accumulator,
    };
  }
}
