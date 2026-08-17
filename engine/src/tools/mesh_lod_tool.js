import { Mesh } from "../renderer/mesh.js";
import { ResourceCache } from "../renderer/resource_cache.js";
import { CacheTypes } from "../renderer/renderer_types.js";
import { log, warn } from "../utility/logging.js";
import { DevConsoleTool } from "./dev_console_tool.js";

const help_text = "Usage: mesh_lod [<lod>|reset]";

export class MeshLODTool extends DevConsoleTool {
  execute(args) {
    if (args.length === 0) {
      log(`Default mesh LOD: ${Mesh.default_min_lod}. ${help_text}`);
      return;
    }

    const argument = args[0].toLowerCase();
    const lod = argument === "reset" ? 0 : Number(argument);
    if (!Number.isInteger(lod) || lod < 0) {
      warn(`Invalid mesh LOD '${args[0]}'. ${help_text}`);
      return;
    }

    Mesh.set_default_min_lod(lod);

    let mesh_count = 0;
    for (const mesh of ResourceCache.get().fetch_all(CacheTypes.MESH).values()) {
      if (!mesh || typeof mesh.set_min_lod !== "function") {
        continue;
      }
      mesh.set_min_lod(lod);
      mesh_count++;
    }

    log(`Default mesh LOD set to ${lod}; updated ${mesh_count} loaded meshes.`);
  }
}
