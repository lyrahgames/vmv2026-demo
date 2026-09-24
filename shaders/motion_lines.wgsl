// Rendering for the GPU-generated, adaptively sampled motion-line bundle.
//
// A trajectory is one continuous triangle strip: every sample contributes
// exactly two vertices. Teaser and dashed strokes use a screen-space offset;
// the full-trajectory stroke uses the sampled surface normal to form a 3D
// ribbon in the surface tangent plane.

struct MotionLineParams {
  seed_count:          u32,
  sample_count:        u32,
  output_stride:       u32,
  vertex_word_stride:  u32,
  palette_stride:      u32,
  morph_weight_stride: u32,
  max_subdivisions:    u32,
  samples_per_second:  f32,
  duration:            f32,
  tolerance:           f32,
  reserved0:           f32,
  reserved1:           f32,
  reserved2:           f32,
  padding0:            u32,
  padding1:            u32,
  padding2:            u32,
};

struct MotionLineStyle {
  // now, visible tail duration, characteristic length, reserved
  timing:   vec4<f32>,
  // strip width (pixels for teaser/dashed, world units for full trajectory),
  // maximum halo depth displacement, style flag (0 teaser, 1 dashed,
  // 2 full trajectory), reserved
  widths:   vec4<f32>,
  // physical viewport width and height in pixels
  viewport: vec4<f32>,
};

struct Uniforms { view_proj: mat4x4<f32>, camera: vec4<f32>, encode_srgb: vec4<f32> };
struct TrajectorySample {
  position: vec4<f32>,
  velocity: vec4<f32>,
  normal: vec4<f32>,
  metadata: vec4<f32>, // x = time, y = arc length
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> line_trajectories: array<TrajectorySample>;
@group(1) @binding(1) var<uniform> line_params: MotionLineParams;
@group(1) @binding(2) var<storage, read> line_counts: array<u32>;
@group(1) @binding(3) var<uniform> line_style: MotionLineStyle;

struct LineOut {
  @builtin(position) position: vec4<f32>,
  // Signed width coordinate: -1 and +1 are the two outer edges. The
  // fragment stage computes v = abs(width_coordinate), so v=0 is the
  // centerline and v>0.5 is exactly the halo region.
  @location(0) width_coordinate: f32,
  @location(1) time: f32,
  @location(2) arc: f32,
  @location(3) speed: f32,
  @location(4) valid: f32,
  // Cumulative trajectory length, retained for styles such as dashes whose
  // phase is anchored to the complete stroke rather than the moving seed.
  @location(5) cumulative_arc: f32,
};

struct LineFragmentOut {
  // Target 0 stores premultiplied color and its weighted opacity sum. Target
  // 1 stores revealage, so overlapping speedlines do not depend on draw order.
  @location(0) accumulation: vec4<f32>,
  @location(1) revealage: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

fn sample_index(seed: u32, sample: u32) -> u32 {
  return seed * line_params.output_stride + sample;
}

fn projected_position(sample: TrajectorySample) -> vec4<f32> {
  return uniforms.view_proj * sample.position;
}

fn projected_xy(clip: vec4<f32>) -> vec2<f32> {
  return clip.xy / max(abs(clip.w), 1.0e-6);
}

fn ribbon_direction(normal_value: vec3<f32>, tangent_value: vec3<f32>) -> vec3<f32> {
  var normal = normal_value;
  if (dot(normal, normal) < 1.0e-8) {
    normal = vec3<f32>(0.0, 1.0, 0.0);
  } else {
    normal = normalize(normal);
  }

  var side = vec3<f32>(0.0);
  if (dot(tangent_value, tangent_value) > 1.0e-8) {
    side = cross(normal, normalize(tangent_value));
  }
  if (dot(side, side) < 1.0e-8) {
    // Motion parallel to the normal has no unique ribbon side. Choose a
    // stable axis in the tangent plane without dividing by a near-zero cross.
    let reference = select(
      vec3<f32>(0.0, 1.0, 0.0),
      vec3<f32>(1.0, 0.0, 0.0),
      abs(normal.y) > 0.9,
    );
    side = cross(normal, reference);
  }
  return normalize(side);
}

// The active colormap from the teaser is retained rather than replaced by a
// flat color: time along the visible tail is the main cue for speedline age.
fn colormap_f1(x: f32) -> f32 {
  return -510.0 * x + 255.0;
}

fn colormap_f2(x: f32) -> f32 {
  return (-1891.7 * x + 217.46) * x + 255.0;
}

fn colormap_f3(x: f32) -> f32 {
  return 9.26643676359015e1 * sin((x - 4.83450094847127e-1) * 9.93)
    + 1.35940451627965e2;
}

fn colormap_f4(x: f32) -> f32 {
  return -510.0 * x + 510.0;
}

fn colormap_f5(x: f32) -> f32 {
  let xx = x - 197169.0 / 251000.0;
  return (2510.0 * xx - 538.31) * xx;
}

fn colormap_red(x: f32) -> f32 {
  if (x < 0.0) {
    return 1.0;
  }
  if (x < 10873.0 / 94585.0) {
    let xx = colormap_f2(x);
    if (xx > 255.0) {
      return (510.0 - xx) / 255.0;
    }
    return xx / 255.0;
  }
  if (x < 0.5) {
    return 1.0;
  }
  if (x < 146169.0 / 251000.0) {
    return colormap_f4(x) / 255.0;
  }
  if (x < 197169.0 / 251000.0) {
    return colormap_f5(x) / 255.0;
  }
  return 0.0;
}

fn colormap_green(x: f32) -> f32 {
  if (x < 10873.0 / 94585.0) {
    return 1.0;
  }
  if (x < 36373.0 / 94585.0) {
    return colormap_f2(x) / 255.0;
  }
  if (x < 0.5) {
    return colormap_f1(x) / 255.0;
  }
  if (x < 197169.0 / 251000.0) {
    return 0.0;
  }
  if (x <= 1.0) {
    return abs(colormap_f5(x)) / 255.0;
  }
  return 0.0;
}

fn colormap_blue(x: f32) -> f32 {
  if (x < 0.0) {
    return 0.0;
  }
  if (x < 36373.0 / 94585.0) {
    return colormap_f1(x) / 255.0;
  }
  if (x < 146169.0 / 251000.0) {
    return colormap_f3(x) / 255.0;
  }
  if (x <= 1.0) {
    return colormap_f4(x) / 255.0;
  }
  return 0.0;
}

fn colormap(x: f32) -> vec3<f32> {
  return vec3<f32>(
    clamp(colormap_red(x), 0.0, 1.0),
    clamp(colormap_green(x), 0.0, 1.0),
    clamp(colormap_blue(x), 0.0, 1.0),
  );
}

// Return the cumulative arc coordinate at the currently active seed. The
// post-process pass stores samples in increasing animation time and increasing
// arc length, while the original teaser stores the envelope coordinate in the
// opposite direction:
//
//   arc_from_seed = arc(seed_at_now) - arc(sample)
//
// In particular, it must be zero at the seed. Using the final animation
// sample here is tempting, but makes a line that is already fully formed at
// the seed for every frame except the end of the animation.
fn arc_at_time(seed: u32, count: u32, time: f32) -> f32 {
  let first = line_trajectories[sample_index(seed, 0u)];
  if (time <= first.metadata.x || count <= 1u) {
    return first.metadata.y;
  }

  let last_index = count - 1u;
  let last = line_trajectories[sample_index(seed, last_index)];
  if (time >= last.metadata.x) {
    return last.metadata.y;
  }

  // Times are monotonic, so a short binary search avoids scanning a complete
  // trajectory for every strip vertex. Interpolating the two enclosing arc
  // values also keeps the envelope smooth when `now` lies between samples.
  var lower = 0u;
  var upper = last_index;
  loop {
    if (upper - lower <= 1u) {
      break;
    }
    let middle = (lower + upper) / 2u;
    let middle_sample = line_trajectories[sample_index(seed, middle)];
    if (middle_sample.metadata.x <= time) {
      lower = middle;
    } else {
      upper = middle;
    }
  }

  let lower_sample = line_trajectories[sample_index(seed, lower)];
  let upper_sample = line_trajectories[sample_index(seed, upper)];
  let interval = max(upper_sample.metadata.x - lower_sample.metadata.x, 1.0e-6);
  let factor = clamp((time - lower_sample.metadata.x) / interval, 0.0, 1.0);
  return mix(lower_sample.metadata.y, upper_sample.metadata.y, factor);
}

// This is the teaser's temporal stroke weight. It is used both for the
// triangle-strip width and for the fragment opacity. Consequently the stroke
// grows out of the seed instead of appearing fully formed at the seed.
fn temporal_weight(time: f32, arc: f32, speed: f32) -> f32 {
  let delta = max(line_style.timing.y, 1.0e-4);
  let age = line_style.timing.x - time;
  if (age < 0.0 || age > delta) {
    return 0.0;
  }
  let characteristic_length = max(line_style.timing.z, 1.0e-4);
  let begin_mask = smoothstep(0.0, 0.3, max(arc, 0.0) / characteristic_length);
  let end_mask = 1.0 - smoothstep(0.95 * delta, delta, age);
  let decay_mask = exp(-2.0 * age / delta);
  let speed_value = speed * delta / characteristic_length;
  let speed_mask = 1.0 - exp(-0.2 * speed_value * speed_value);
  return begin_mask * end_mask * decay_mask * speed_mask;
}

// Geometry weight from paper-compasso's bundle vertex shader. The Compasso
// dash and speed masks belong exclusively to its fragment shader.
fn compasso_geometry_weight(time: f32, arc: f32) -> f32 {
  let delta = max(line_style.timing.y, 1.0e-4);
  let age = line_style.timing.x - time;
  if (age < 0.0 || age > delta) {
    return 0.0;
  }
  let characteristic_length = max(line_style.timing.z, 1.0e-4);
  let begin_mask = smoothstep(0.02, 0.05, max(arc, 0.0) / characteristic_length);
  let end_mask = 1.0 - smoothstep(0.95 * delta, delta, age);
  return begin_mask * end_mask * exp(-2.0 * age / delta);
}

@vertex
fn line_vertex(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) seed_index: u32,
) -> LineOut {
  var output: LineOut;
  let count = max(line_counts[seed_index], 1u);
  let last = count - 1u;
  let requested_sample = vertex_index / 2u;
  let sample = min(requested_sample, last);
  let use_positive_side = vertex_index % 2u == 1u;
  let valid = f32(requested_sample < count);

  // Invalid fixed-capacity tail samples use the final real sample and its
  // final join tangent. They therefore duplicate the last pair exactly and
  // cannot create a sliver while their valid flag discards their fragments.
  var previous = sample;
  var next = sample;
  if (sample > 0u) {
    previous = sample - 1u;
  }
  if (sample + 1u < count) {
    next = sample + 1u;
  }

  let previous_sample = line_trajectories[sample_index(seed_index, previous)];
  let current_sample = line_trajectories[sample_index(seed_index, sample)];
  let next_sample = line_trajectories[sample_index(seed_index, next)];
  let previous_clip = projected_position(previous_sample);
  let current_clip = projected_position(current_sample);
  let next_clip = projected_position(next_sample);
  let previous_xy = projected_xy(previous_clip);
  let current_xy = projected_xy(current_clip);
  let next_xy = projected_xy(next_clip);

  // The trajectory is stored from the beginning to the end of the animation,
  // but the teaser's `begin_mask` is measured from the moving seed backwards
  // along the visible tail. Reconstruct that coordinate at the current `now`
  // rather than measuring against the animation's final sample.
  let seed_arc = arc_at_time(seed_index, count, line_style.timing.x);
  let arc = max(seed_arc - current_sample.metadata.y, 0.0);

  let viewport = max(line_style.viewport.xy, vec2<f32>(1.0, 1.0));
  var tangent_previous = (current_xy - previous_xy) * viewport;
  var tangent_next = (next_xy - current_xy) * viewport;
  let projected_motion = max(length(tangent_previous), length(tangent_next));

  if (dot(tangent_previous, tangent_previous) < 1.0e-8) {
    tangent_previous = tangent_next;
  }
  if (dot(tangent_next, tangent_next) < 1.0e-8) {
    tangent_next = tangent_previous;
  }
  if (dot(tangent_previous, tangent_previous) < 1.0e-8) {
    tangent_previous = vec2<f32>(1.0, 0.0);
  }
  if (dot(tangent_next, tangent_next) < 1.0e-8) {
    tangent_next = vec2<f32>(1.0, 0.0);
  }

  // The depth-halo construction uses ||V x D|| as the perspective factor:
  // a trajectory parallel to the viewing ray has no well-defined visible
  // side, so its strip width must collapse instead of being amplified by a
  // nearly singular perpendicular. The screen-space normal still supplies
  // the actual extrusion direction, which keeps the strip view aligned.
  let world_tangent = next_sample.position.xyz - previous_sample.position.xyz;
  let world_tangent_length = length(world_tangent);
  let to_camera = uniforms.camera.xyz - current_sample.position.xyz;
  let to_camera_length = length(to_camera);
  var perspective_factor = 0.0;
  if (world_tangent_length > 1.0e-6 && to_camera_length > 1.0e-6) {
    perspective_factor = length(cross(
      world_tangent / world_tangent_length,
      to_camera / to_camera_length,
    ));
  }

  var extrusion = vec2<f32>(0.0, 0.0);
  if (projected_motion > 1.0e-6) {
    let direction_previous = normalize(tangent_previous);
    let direction_next = normalize(tangent_next);
    var average_direction = direction_previous + direction_next;
    if (dot(average_direction, average_direction) < 1.0e-8) {
      average_direction = direction_next;
    }
    let average = normalize(average_direction);
    let screen_normal = vec2<f32>(-average.y, average.x);
    extrusion = screen_normal * perspective_factor;
  }
  var side = -1.0;
  if (use_positive_side) {
    side = 1.0;
  }
  let speed = max(
    (next_sample.metadata.y - previous_sample.metadata.y)
      / max(next_sample.metadata.x - previous_sample.metadata.x, 1.0e-6),
    0.0,
  );
  let compasso_style = line_style.widths.z > 0.5;
  let stroke_weight = select(
    temporal_weight(current_sample.metadata.x, arc, speed),
    compasso_geometry_weight(current_sample.metadata.x, arc),
    compasso_style,
  );
  // Compasso emits an outer extent at +/-1.5 line widths and interpolates
  // its fragment coordinate across +/-3. The teaser keeps its existing
  // +/-0.5-width strip and +/-1 coordinate exactly.
  let half_extent = select(0.5, 1.5, compasso_style);
  let pixel_offset = side * line_style.widths.x * half_extent * extrusion * stroke_weight;
  let ndc_offset = 2.0 * pixel_offset / viewport;

  var clip = current_clip;
  clip.x += ndc_offset.x * clip.w;
  clip.y += ndc_offset.y * clip.w;
  output.position = clip;
  output.width_coordinate = side * select(1.0, 3.0, compasso_style);
  output.time = current_sample.metadata.x;
  output.arc = arc;
  output.speed = speed;
  output.valid = valid;
  output.cumulative_arc = current_sample.metadata.y;
  return output;
}

@vertex
fn full_trajectory_vertex(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) seed_index: u32,
) -> LineOut {
  var output: LineOut;
  let count = max(line_counts[seed_index], 1u);
  let last = count - 1u;
  let requested_sample = vertex_index / 2u;
  let sample = min(requested_sample, last);
  var previous = sample;
  var next = sample;
  if (sample > 0u) {
    previous = sample - 1u;
  }
  if (sample < last) {
    next = sample + 1u;
  }
  let previous_sample = line_trajectories[sample_index(seed_index, previous)];
  let current_sample = line_trajectories[sample_index(seed_index, sample)];
  let next_sample = line_trajectories[sample_index(seed_index, next)];

  // The post-process stores the Catmull-Rom derivative at this exact sample.
  // Neighboring polyline differences are biased when adjacent segments use
  // different subdivision counts and make a wide ribbon look faceted.
  var tangent = current_sample.velocity.xyz;
  if (dot(tangent, tangent) < 1.0e-8) {
    tangent = next_sample.position.xyz - previous_sample.position.xyz;
  }
  if (dot(tangent, tangent) < 1.0e-8) {
    tangent = next_sample.position.xyz - current_sample.position.xyz;
  }
  if (dot(tangent, tangent) < 1.0e-8) {
    tangent = current_sample.position.xyz - previous_sample.position.xyz;
  }
  let side = select(-1.0, 1.0, vertex_index % 2u == 1u);
  let direction = ribbon_direction(current_sample.normal.xyz, tangent);
  let ribbon_position = current_sample.position.xyz
    + side * 0.5 * line_style.widths.x * direction;

  output.position = uniforms.view_proj * vec4<f32>(ribbon_position, 1.0);
  output.width_coordinate = side;
  output.time = current_sample.metadata.x;
  output.arc = 0.0;
  output.speed = 0.0;
  output.valid = f32(requested_sample < count);
  output.cumulative_arc = current_sample.metadata.y;
  return output;
}

@fragment
fn line_fragment(input: LineOut) -> LineFragmentOut {
  if (input.valid < 0.5) {
    discard;
  }

  let delta = max(line_style.timing.y, 1.0e-4);
  var age = line_style.timing.x - input.time;
  if (age < 0.0 || age > delta) {
    discard;
  }

  let weight = temporal_weight(input.time, input.arc, input.speed);
  if (weight <= 1.0e-4) {
    discard;
  }

  let color_a = colormap(age / delta / 0.8 + 0.2);
  let color_b = colormap(age / delta / 0.9 + 0.1);
  let v = abs(input.width_coordinate);
  var line_color = color_a;
  if (v > 0.5) {
    // Keep the temporal color only as a narrow contour at the core/halo
    // boundary. Most of the halo is the characteristic opaque white band.
    let halo_mix = smoothstep(0.5, 0.68, v);
    line_color = mix(color_a, vec3<f32>(1.0), halo_mix);
  } else {
    // Keep the original direction: the midpoint is color_b and the inner
    // edge is color_a, rather than reversing the stroke UV.
    let inner_mix = smoothstep(0.3125, 0.5, v);
    line_color = mix(color_b, color_a, inner_mix);
  }

  var output: LineFragmentOut;
  // Restore the teaser's temporal opacity. The color target uses weighted
  // blending, so this alpha is accumulated without relying on draw order.
  let alpha = clamp(weight, 0.0, 1.0);
  let oit_weight = clamp(0.01 + 4.0 * alpha, 0.01, 8.0);
  output.accumulation = vec4<f32>(line_color * alpha * oit_weight, alpha * oit_weight);
  // Only the source alpha is used by the revealage target's blend state.
  output.revealage = vec4<f32>(0.0, 0.0, 0.0, alpha);
  // v>0.5 is the halo. Its depth is displaced linearly with the distance
  // from the centerline. It remains depth-tested before OIT accumulation.
  var depth = input.position.z;
  if (v > 0.5) {
    // d_new = d_old + d_max * v. The core remains at its original depth;
    // only the halo writes the displaced value.
    depth += line_style.widths.y * v;
  }
  output.depth = clamp(depth, 0.0, 1.0);
  return output;
}

// An opaque dark gray stroke for the complete extracted trajectory.
@fragment
fn full_trajectory_fragment(input: LineOut) -> LineFragmentOut {
  if (input.valid < 0.5) {
    discard;
  }

  var output: LineFragmentOut;
  output.accumulation = vec4<f32>(0.08, 0.08, 0.08, 1.0);
  output.revealage = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  output.depth = clamp(input.position.z, 0.0, 1.0);
  return output;
}

@group(1) @binding(0) var oit_accumulation: texture_2d<f32>;
@group(1) @binding(1) var oit_revealage: texture_2d<f32>;

struct CompositeOut {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn oit_composite_vertex(@builtin(vertex_index) vertex_index: u32) -> CompositeOut {
  // One oversized triangle covers the complete render target without a
  // vertex buffer or interpolated UV seam.
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  var output: CompositeOut;
  output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
  return output;
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
  let low = color * 12.92;
  let high = 1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
  return select(low, high, color > vec3<f32>(0.0031308));
}

@fragment
fn oit_composite_fragment(input: CompositeOut) -> @location(0) vec4<f32> {
  let pixel = vec2<i32>(i32(input.position.x), i32(input.position.y));
  let accumulation = textureLoad(oit_accumulation, pixel, 0);
  let revealage = textureLoad(oit_revealage, pixel, 0).r;
  let alpha = clamp(1.0 - revealage, 0.0, 1.0);
  if (alpha <= 1.0e-5 || accumulation.a <= 1.0e-5) {
    return vec4<f32>(0.0);
  }
  let color = accumulation.rgb / accumulation.a;
  let output_color = mix(color, linear_to_srgb(color), uniforms.encode_srgb.x);
  return vec4<f32>(output_color, alpha);
}
