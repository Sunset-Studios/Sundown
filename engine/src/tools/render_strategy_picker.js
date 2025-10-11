import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";
import { RenderStrategyType } from "../renderer/renderer_types.js";

const deferred = "deferred";
const pathtracing = "pathtracing";
const path_tracing = "path-tracing";
const pt = "pt";

const help_text = "Available strategies: deferred, pathtracing (or pt)";

/**
 * RenderStrategyPicker allows the user to switch between rendering strategies from the command line.
 */
export class RenderStrategyPicker extends DevConsoleTool {
  execute(args) {
    const strategy_name = args[0];
    if (strategy_name) {
      switch (strategy_name) {
        case deferred:
          Renderer.get().set_render_strategy_type(RenderStrategyType.Deferred);
          break;
        case pathtracing:
        case path_tracing:
        case pt:
          Renderer.get().set_render_strategy_type(RenderStrategyType.PathTracing);
          break;
        default:
          console.log(help_text);
          break;
      }
    }
  }
}

