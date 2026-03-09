import { DevConsoleTool } from "./dev_console_tool.js";
import { Renderer } from "../renderer/renderer.js";
import { AOStrategyType } from "../renderer/renderer_types.js";

const vbao = "vbao";
const rtao = "rtao";

const help_text = "Available strategies: vbao, rtao";

export class AOStrategyPicker extends DevConsoleTool {
  execute(args) {
    const strategy_name = args[0];
    if (strategy_name) {
      switch (strategy_name) {
        case vbao:
          Renderer.get().set_ao_strategy_type(AOStrategyType.VBAO);
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
