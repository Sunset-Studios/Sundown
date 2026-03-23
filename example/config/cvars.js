import { EngineCVars } from "../../engine/config/cvars.js";
import { is_mobile_platform } from "../../engine/src/utility/platform.js";

export default ({ project }) => {
  const profile = is_mobile_platform() ? "mobile" : "desktop";

  const base = {
    [EngineCVars.Renderer.DebugDraw]: "none",
    [EngineCVars.Renderer.DebugTextureLevel]: 0,
  };

  if (profile === "mobile") {
    return {
      name: `${project?.name || "project"}_cvars`,
      cvars: {
        ...base,
        [EngineCVars.Renderer.RenderStrategy]: "deferred",
        [EngineCVars.Renderer.ShadowsEnabled]: true,
        [EngineCVars.Renderer.GIEnabled]: false,
        [EngineCVars.Renderer.AOEnabled]: true,
        [EngineCVars.Renderer.ReflectionsEnabled]: false,
      },
    };
  }

  return {
    name: `${project?.name || "project"}_cvars`,
    cvars: {
      ...base,
      [EngineCVars.Renderer.RenderStrategy]: "deferred",
      [EngineCVars.Renderer.ShadowsEnabled]: true,
      [EngineCVars.Renderer.GIEnabled]: true,
      [EngineCVars.Renderer.AOEnabled]: true,
      [EngineCVars.Renderer.ReflectionsEnabled]: true,
    },
  };
};
