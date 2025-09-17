export function frame_runner(frame_callback, max_fps = 60) {
  const fps = 1000 / (max_fps || 60);

  var delta_time = 0;
  var previous_time = performance.now();

  return (async function loop(timestamp) {
    delta_time = timestamp - previous_time;
    
    if (delta_time > fps) {
      previous_time = timestamp - (delta_time % fps);
      await frame_callback(delta_time / 1000.0);
    }

    requestAnimationFrame(loop);
  })();
}
