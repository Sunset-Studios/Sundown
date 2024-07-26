export function frame_runner(frame_callback, max_fps = 60) {
  const fps = 1000 / (max_fps || 60);

  var current_time = 0;
  var delta_time = 0;
  var previous_time = performance.now();

  return (function loop() {
    requestAnimationFrame(loop);

    current_time = performance.now();
    delta_time = current_time - previous_time;

    if (delta_time > fps) {
      previous_time = current_time - (delta_time % fps);
      frame_callback(delta_time / 1000.0);
    }
  })();
}
