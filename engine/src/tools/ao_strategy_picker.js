import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";
import { AOStrategyType } from "../renderer/renderer_types.js";

const gtao = "gtao";
const rtao = "rtao";

const help_text = "Available strategies: gtao, rtao";

export class AOStrategyPicker extends DevConsoleTool {
  execute(args) {
    const strategy_name = args[0];
    if (strategy_name) {
      switch (strategy_name) {
        case gtao:
          Renderer.get().set_ao_strategy_type(AOStrategyType.GTAO);
          break;
        case rtao:
          Renderer.get().set_ao_strategy_type(AOStrategyType.RTAO);
          break;
        default:
          console.log(help_text);
          break;
      }
    }
  }
}
