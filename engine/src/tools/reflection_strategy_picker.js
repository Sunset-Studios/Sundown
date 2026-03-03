import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";
import { ReflectionStrategyType } from "../renderer/renderer_types.js";

const ssr = "ssr";

const help_text = "Available strategies: ssr";

export class ReflectionStrategyPicker extends DevConsoleTool {
  execute(args) {
    const strategy_name = args[0];
    if (strategy_name) {
      switch (strategy_name) {
        case ssr:
          Renderer.get().set_reflection_strategy_type(ReflectionStrategyType.SSR);
          break;
        default:
          console.log(help_text);
          break;
      }
    }
  }
}
