import { EngineCVars } from "../../engine/config/cvars.js";

const example_cvar_config = ({ project }) => ({
  name: `${project?.name || "example"}_cvars`,
  cvars: {
    [EngineCVars.Renderer.RenderStrategy]: "deferred",
    [EngineCVars.Renderer.ShadowsEnabled]: true,
    [EngineCVars.Renderer.GIEnabled]: true,
    [EngineCVars.Renderer.AOEnabled]: true,
  },
});

export default example_cvar_config;
