import { SharedFrameInfoBuffer, SharedViewBuffer } from "../core/shared_data.js";
import { InputProvider } from "../input/input_provider.js";
import { InputKey } from "../input/input_types.js";
import { MeshTaskQueue } from "../renderer/mesh_task_queue.js";
import { Renderer } from "../renderer/renderer.js";
import { panel, label } from "../ui/2d/immediate.js";
import { DevConsoleTool } from "./dev_console_tool.js";

const stats_panel_config = {
  layout: "column",
  gap: 4,
  y: 25,
  x: 25,
  anchor_x: "right",
  dont_consume_cursor_events: true,
  background_color: "rgba(0, 0, 0, 0.7)",
  width: 500,
  height: "75%",
  padding: 10,
  border: "1px solid rgb(68, 68, 68)",
  corner_radius: 5,
  scrollable: true,
  scroll_speed: 35,
  clip: true,
};

const stats_label_config = {
  y: 0,
  text_color: "#fff",
  wrap: true,
  font: "16px monospace",
  width: "100%",
  height: "fit-content",
  text_valign: "middle",
  text_align: "left",
  text_padding: 5,
};

const stats_label_config_small = {
  ...stats_label_config,
  width: "fit-content",
  x: 0,
};

const value_label_config = {
  ...stats_label_config,
  text_color: "#4eaaff",
  width: "fit-content",
  x: 0,
};

const note_label_config = {
  ...stats_label_config,
  text_color: "#b8b8b8",
  font: "14px monospace",
};

function format_number(value) {
  return Number(value ?? 0).toLocaleString();
}

function format_percent(value, total) {
  if (!total) {
    return "0.0%";
  }

  return `${((value / total) * 100).toFixed(1)}%`;
}

function format_count_with_percent(value, total) {
  return `${format_number(value)} (${format_percent(value, total)})`;
}

function stat_row(label_text, value_text, value_config = value_label_config) {
  panel({ layout: "row", gap: 4, width: "100%", height: 25, anchor_x: "left", x: 0, y: 0 }, () => {
    label(`${label_text}:`, stats_label_config_small);
    label(value_text, value_config);
  });
}

function section_header(text) {
  label(text, { ...stats_label_config, font: "18px monospace", text_color: "#0ff" });
  label("--------------------------------", stats_label_config);
}

function get_meshlet_stats() {
  if (!__DEV__) {
    return null;
  }

  const renderer = Renderer.get();
  const render_strategy = renderer?.get_render_strategy?.();
  const culling_pipeline = render_strategy?.culling_pipeline;
  const visibility_buffer_pipeline = render_strategy?.visibility_buffer_pipeline;
  const current_view = SharedFrameInfoBuffer.get_view_index();
  const view_data = SharedViewBuffer.get_view_data(current_view);
  const visibility_buckets = MeshTaskQueue.get_visibility_shader_buckets();
  const total_meshlet_count = MeshTaskQueue.get_total_meshlet_count();
  const culling_stats = culling_pipeline?.get_meshlet_stats?.(current_view) ?? null;
  const frustum_bucket_stats =
    visibility_buffer_pipeline?.get_bucket_meshlet_stats?.(
      current_view,
      visibility_buckets,
      "frustum"
    ) ?? [];
  const occlusion_bucket_stats =
    visibility_buffer_pipeline?.get_bucket_meshlet_stats?.(
      current_view,
      visibility_buckets,
      "occlusion"
    ) ?? [];

  const frustum_compacted_count = frustum_bucket_stats.reduce(
    (sum, bucket) => sum + (bucket.instance_count ?? 0),
    0
  );
  const occlusion_compacted_count = occlusion_bucket_stats.reduce(
    (sum, bucket) => sum + (bucket.instance_count ?? 0),
    0
  );

  return {
    current_view,
    occlusion_enabled: !!view_data?.occlusion_enabled,
    total_meshlet_count,
    visibility_bucket_count: visibility_buckets.length,
    frustum_count: culling_stats?.frustum?.instance_count ?? 0,
    occlusion_count: culling_stats?.occlusion?.instance_count ?? 0,
    frustum_compacted_count,
    occlusion_compacted_count,
    frustum_bucket_stats,
    occlusion_bucket_stats,
  };
}

function render_bucket_stats(title, bucket_stats, source_count) {
  section_header(title);

  if (!bucket_stats.length) {
    label("No visibility buckets registered for this frame.", note_label_config);
    return;
  }

  const active_buckets = bucket_stats.filter((bucket) => bucket.instance_count > 0);
  const inactive_bucket_count = bucket_stats.length - active_buckets.length;

  if (!active_buckets.length) {
    label("No compacted meshlets recorded for this stage.", note_label_config);
  }

  for (const bucket of active_buckets) {
    const bucket_name = bucket.template_name || `Bucket ${bucket.key}`;
    stat_row(bucket_name, format_count_with_percent(bucket.instance_count, source_count));
  }

  if (inactive_bucket_count > 0) {
    label(`${format_number(inactive_bucket_count)} buckets currently at 0 meshlets.`, note_label_config);
  }
}

export class MeshletStats extends DevConsoleTool {
  is_open = false;
  scene = null;

  update() {
    if (!this.is_open) return;
    this.render();
  }

  render() {
    const stats = get_meshlet_stats();

    const panel_state = panel(stats_panel_config, () => {
      section_header("Meshlet Stats");

      if (!stats) {
        label("Meshlet stats are only available in development builds.", note_label_config);
        return;
      }

      label(
        "Readback reflects the most recently completed frame.",
        note_label_config
      );

      stat_row("Current view", format_number(stats.current_view));
      stat_row("Submitted meshlet instances", format_number(stats.total_meshlet_count));
      stat_row("Visibility buckets", format_number(stats.visibility_bucket_count));
      stat_row("After frustum", format_count_with_percent(stats.frustum_count, stats.total_meshlet_count));
      stat_row(
        "After frustum compaction",
        format_count_with_percent(stats.frustum_compacted_count, stats.frustum_count)
      );

      if (stats.occlusion_enabled) {
        stat_row("After occlusion", format_count_with_percent(stats.occlusion_count, stats.frustum_count));
        stat_row(
          "After occlusion compaction",
          format_count_with_percent(stats.occlusion_compacted_count, stats.occlusion_count)
        );
      } else {
        stat_row("Occlusion", "Disabled", { ...value_label_config, text_color: "#ffbf69" });
      }

      render_bucket_stats(
        "Frustum Compaction by Bucket",
        stats.frustum_bucket_stats,
        stats.frustum_count
      );

      render_bucket_stats(
        "Occlusion Compaction by Bucket",
        stats.occlusion_bucket_stats,
        stats.occlusion_count
      );
    });

    if (this.is_open && InputProvider.get_action(InputKey.B_mouse_left)) {
      if (!panel_state.hovered) {
        this.hide();
      }
      InputProvider.consume_action(InputKey.B_mouse_left);
    }
  }

  execute() {
    this.toggle();
  }

  toggle() {
    this.is_open = !this.is_open;
    if (this.scene) {
      if (this.is_open && typeof this.scene.show_dev_cursor === "function") {
        this.scene.show_dev_cursor();
      } else if (!this.is_open && typeof this.scene.hide_dev_cursor === "function") {
        this.scene.hide_dev_cursor();
      }
    }
  }

  show() {
    this.is_open = true;
    if (this.scene && typeof this.scene.show_dev_cursor === "function") {
      this.scene.show_dev_cursor();
    }
  }

  hide() {
    this.is_open = false;
    if (this.scene && typeof this.scene.hide_dev_cursor === "function") {
      this.scene.hide_dev_cursor();
    }
  }

  set_scene(scene) {
    this.scene = scene;
  }
}
