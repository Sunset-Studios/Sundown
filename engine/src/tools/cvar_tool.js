import { CVarSystem, CVarType } from "../core/cvar_system.js";
import { log, warn } from "../utility/logging.js";
import { DevConsoleTool } from "./dev_console_tool.js";

const help_text = "Usage: cvar [list [prefix] | <name> [value|toggle|reset]]";

export class CVarTool extends DevConsoleTool {
  execute(args) {
    if (args.length === 0) {
      this._log_list();
      return;
    }

    if (args[0] === "list") {
      this._log_list(args[1] || "");
      return;
    }

    const name = args[0];
    const definition = CVarSystem.get_definition(name);
    if (!definition) {
      warn(`Unknown cvar: ${name}`);
      this._log_matches(name);
      return;
    }

    if (args.length === 1) {
      log(this._format_entry(name, definition));
      return;
    }

    try {
      const action = args[1].toLowerCase();
      if (action === "toggle") {
        CVarSystem.toggle(name, { source: "console" });
      } else if (action === "reset") {
        CVarSystem.reset(name, { source: "console" });
      } else {
        CVarSystem.set_from_string(name, args.slice(1).join(" "), { source: "console" });
      }

      log(this._format_entry(name, definition));
    } catch (err) {
      warn(`Failed to update cvar '${name}': ${err.message}`);
    }
  }

  _log_list(prefix = "") {
    const entries = CVarSystem.list(prefix);
    if (entries.length === 0) {
      log(prefix ? `No cvars found for prefix '${prefix}'.` : "No cvars registered.");
      return;
    }

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      log(this._format_entry(entry.name, entry.definition));
    }
  }

  _log_matches(fragment) {
    const entries = CVarSystem.list().filter((entry) => entry.name.includes(fragment));
    if (entries.length === 0) {
      log(help_text);
      return;
    }

    for (let i = 0; i < entries.length; i++) {
      log(this._format_entry(entries[i].name, entries[i].definition));
    }
  }

  _format_entry(name, definition) {
    let suffix = ` (${definition.type}`;
    if (definition.type === CVarType.Enum) {
      suffix += `: ${Array.from(definition.label_to_value.keys()).join(", ")}`;
    }
    suffix += ")";

    if (definition.description) {
      suffix += ` - ${definition.description}`;
    }

    return `${name} = ${CVarSystem.format_value(name)}${suffix}`;
  }
}
