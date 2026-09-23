// Dashed motion-line fragment adapted from paper-compasso/shaders/bundle.fs.glsl.
// This source is appended to motion_lines.wgsl at compile time, so it shares
// LineOut, MotionLineStyle, and OIT output types.

// Active Compasso colormap from bundle.fs.glsl (not the teaser colormap).
fn compasso_colormap_red(x: f32) -> f32 {
  if (x < 0.75) {
    return 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
  }
  return (13.0 + 8.0 / 9.0) / 10.0 * x - (3.0 + 8.0 / 9.0) / 10.0;
}

fn compasso_colormap_green(x: f32) -> f32 {
  if (x <= 0.375) {
    return 8.0 / 9.0 * x - (13.0 + 8.0 / 9.0) / 1000.0;
  }
  if (x <= 0.75) {
    return (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 100.0;
  }
  return 8.0 / 9.0 * x + 1.0 / 9.0;
}

fn compasso_colormap_blue(x: f32) -> f32 {
  if (x <= 0.375) {
    return (1.0 + 2.0 / 9.0) * x - (13.0 + 8.0 / 9.0) / 1000.0;
  }
  return 8.0 / 9.0 * x + 1.0 / 9.0;
}

fn compasso_colormap(x: f32) -> vec3<f32> {
  return clamp(vec3<f32>(
    compasso_colormap_red(x),
    compasso_colormap_green(x),
    compasso_colormap_blue(x),
  ), vec3<f32>(0.0), vec3<f32>(1.0));
}

// `bundle.fs.glsl` used a slightly stronger speed falloff for its dash style
// than the teaser stroke. Keep that style-specific 0.3 coefficient here.
fn dashed_temporal_weight(time: f32, arc: f32, speed: f32) -> f32 {
  let delta = max(line_style.timing.y, 1.0e-4);
  let age = line_style.timing.x - time;
  let characteristic_length = max(line_style.timing.z, 1.0e-4);
  let begin_mask = smoothstep(0.0, 0.3, max(arc, 0.0) / characteristic_length);
  let end_mask = 1.0 - smoothstep(0.95 * delta, delta, age);
  let decay_mask = exp(-2.0 * age / delta);
  let speed_value = speed * delta / characteristic_length;
  let speed_mask = 1.0 - exp(-0.3 * speed_value * speed_value);
  return begin_mask * end_mask * decay_mask * speed_mask;
}

@fragment
fn dashed_line_fragment(input: LineOut) -> LineFragmentOut {
  if (input.valid < 0.5) {
    discard;
  }

  let delta = max(line_style.timing.y, 1.0e-4);
  let age = line_style.timing.x - input.time;
  if (age < 0.0 || age > delta) {
    discard;
  }

  // Keep the Compasso dash frequency and soft 0.35--0.45 edge. The old
  // shader called this `varc`: the cumulative arc length of a trajectory.
  let characteristic_length = max(line_style.timing.z, 1.0e-4);
  let dash_u = abs(2.0 * fract(10.0 * input.cumulative_arc / characteristic_length) - 1.0);
  let dash_mask = smoothstep(0.35, 0.45, dash_u);
  let weight = dashed_temporal_weight(input.time, input.arc, input.speed) * dash_mask;
  if (weight <= 1.0e-4) {
    discard;
  }

  let color_a = compasso_colormap(age / delta / 0.8 + 0.2);
  let v = abs(input.width_coordinate);
  var line_color: vec3<f32>;
  if (v >= 0.8) {
    // Exact active Compasso outer band: temporal color at the inner edge,
    // fading to white across the outer half-width.
    let edge_mix = smoothstep(0.8, 1.1, v);
    line_color = mix(vec3<f32>(1.0), color_a, 1.0 - edge_mix);
  } else {
    let inner_mix = smoothstep(0.5, 0.8, v);
    let begin_color = compasso_colormap(age / delta / 0.9 + 0.1);
    let end_color = compasso_colormap(age / delta / 0.6 + 0.4);
    line_color = mix(begin_color, end_color, 1.0 - inner_mix);
  }

  let alpha = clamp(weight, 0.0, 1.0);
  let oit_weight = clamp(0.01 + 4.0 * alpha, 0.01, 8.0);
  var output: LineFragmentOut;
  output.accumulation = vec4<f32>(line_color * alpha * oit_weight, alpha * oit_weight);
  output.revealage = vec4<f32>(0.0, 0.0, 0.0, alpha);
  // The Compasso shader used only a tiny geometry-side depth separation;
  // unlike the teaser, it does not apply a fragment depth halo.
  output.depth = clamp(input.position.z, 0.0, 1.0);
  return output;
}
