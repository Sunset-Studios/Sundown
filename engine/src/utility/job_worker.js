import { register_job_handler, start_job_worker } from "./job_worker_runtime.js";
import { load_texture_bitmaps_job } from "./texture_load_job.js";

register_job_handler("load_texture_bitmaps", load_texture_bitmaps_job);

start_job_worker();
