// Adaptive GPU post-process for uniformly traced trajectories.
//
// One invocation owns one seed and writes a compact prefix of the fixed
// output stride. It tests each Catmull–Rom segment against its straight chord;
// every segment receives finer spline samples, and curved segments receive
// further dyadic subdivisions. Counts make the compact prefix available to
// the render and future filtering/styling passes.

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

struct TrajectorySample {
  position: vec4<f32>,
  velocity: vec4<f32>,
  normal: vec4<f32>,
  metadata: vec4<f32>, // x = time, y = arc length
};

@group(0) @binding(0) var<storage, read> raw_trajectories: array<TrajectorySample>;
@group(0) @binding(1) var<storage, read_write> trajectories: array<TrajectorySample>;
@group(0) @binding(2) var<storage, read_write> trajectory_counts: array<u32>;
@group(0) @binding(3) var<uniform> params: MotionLineParams;

fn raw_sample(seed: u32, sample: u32) -> TrajectorySample {
  return raw_trajectories[seed * params.sample_count + sample];
}

fn control_before(seed: u32, segment: u32) -> vec3<f32> {
  let p1 = raw_sample(seed, segment).position.xyz;
  if (segment == 0u) {
    // Extrapolating the first tangent keeps a genuinely linear trajectory
    // linear at the endpoint, so the adaptive test does not add artificial
    // samples merely because a control point was duplicated.
    let p2 = raw_sample(seed, 1u).position.xyz;
    return 2.0 * p1 - p2;
  }
  return raw_sample(seed, segment - 1u).position.xyz;
}

fn control_after(seed: u32, segment: u32, last: u32) -> vec3<f32> {
  let p1 = raw_sample(seed, segment).position.xyz;
  let p2 = raw_sample(seed, segment + 1u).position.xyz;
  if (segment + 2u > last) {
    return 2.0 * p2 - p1;
  }
  return raw_sample(seed, segment + 2u).position.xyz;
}

fn catmull_rom(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (
    (2.0 * p1) +
    (-p0 + p2) * t +
    (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
    (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
  );
}

fn catmull_rom_derivative(
  p0: vec3<f32>,
  p1: vec3<f32>,
  p2: vec3<f32>,
  p3: vec3<f32>,
  t: f32,
) -> vec3<f32> {
  let t2 = t * t;
  return 0.5 * (
    (-p0 + p2) +
    2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t +
    3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t2
  );
}

fn safe_normalize(value: vec3<f32>) -> vec3<f32> {
  if (dot(value, value) <= 0.00000001) {
    return vec3<f32>(0.0, 1.0, 0.0);
  }
  return normalize(value);
}

fn segment_deviation(
  p0: vec3<f32>,
  p1: vec3<f32>,
  p2: vec3<f32>,
  p3: vec3<f32>,
) -> f32 {
  var deviation = 0.0;
  // Three interior probes are inexpensive and catch the usual extrema of a
  // cubic without making the adaptive decision recursive.
  for (var probe = 1u; probe <= 3u; probe = probe + 1u) {
    let t = f32(probe) * 0.25;
    let curve = catmull_rom(p0, p1, p2, p3, t);
    let chord = mix(p1, p2, t);
    deviation = max(deviation, distance(curve, chord));
  }
  return deviation;
}

fn segment_steps(
  p0: vec3<f32>,
  p1: vec3<f32>,
  p2: vec3<f32>,
  p3: vec3<f32>,
) -> u32 {
  let deviation = segment_deviation(p0, p1, p2, p3);
  // A chord may fall within the geometric tolerance while its two joins
  // still look angular, especially under a wide ribbon. Always sample the
  // Catmull-Rom curve more finely than the original animation FPS.
  var steps = min(params.max_subdivisions, 8u);
  // The chord error shrinks quadratically for this dyadic subdivision scheme.
  // The flatness test adds a second level only for the sharpest bends.
  loop {
    if (steps >= params.max_subdivisions ||
        deviation / f32(steps * steps) <= params.tolerance) {
      break;
    }
    steps = steps * 2u;
  }
  return steps;
}

fn write_sample(
  seed: u32,
  output_index: u32,
  position: vec3<f32>,
  velocity: vec3<f32>,
  normal: vec3<f32>,
  time: f32,
  arc_length: f32,
) {
  // Each seed owns an independent fixed-size slice. The local output index
  // is intentionally compact, but the storage address must include the
  // seed-major stride or all seeds would overwrite the first trajectory.
  trajectories[seed * params.output_stride + output_index] = TrajectorySample(
    vec4<f32>(position, 1.0),
    vec4<f32>(velocity, 0.0),
    vec4<f32>(safe_normalize(normal), 0.0),
    vec4<f32>(time, arc_length, 0.0, 0.0),
  );
}

// A modest workgroup size keeps the post pass within the baseline dispatch
// limit even when "all vertices" selects a large surface mesh.
@compute @workgroup_size(64, 1, 1)
fn postprocess(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let seed = invocation.x;
  if (seed >= params.seed_count) {
    return;
  }

  let last = params.sample_count - 1u;
  var output_index = 0u;
  var arc_length = 0.0;
  var previous_position = raw_sample(seed, 0u).position.xyz;

  if (params.sample_count > 1u) {
    for (var segment = 0u; segment < last; segment = segment + 1u) {
      let p1_sample = raw_sample(seed, segment);
      let p2_sample = raw_sample(seed, segment + 1u);
      let p0 = control_before(seed, segment);
      let p1 = p1_sample.position.xyz;
      let p2 = p2_sample.position.xyz;
      let p3 = control_after(seed, segment, last);
      let steps = segment_steps(p0, p1, p2, p3);
      let time0 = p1_sample.metadata.x;
      let time1 = p2_sample.metadata.x;

      for (var local = 0u; local < steps; local = local + 1u) {
        let t = f32(local) / f32(steps);
        let position = catmull_rom(p0, p1, p2, p3, t);
        if (output_index > 0u) {
          arc_length = arc_length + distance(previous_position, position);
        }
        let time = mix(time0, time1, t);
        let velocity = catmull_rom_derivative(p0, p1, p2, p3, t) /
          max(time1 - time0, 0.000001);
        let normal = mix(p1_sample.normal.xyz, p2_sample.normal.xyz, t);
        write_sample(seed, output_index, position, velocity, normal, time, arc_length);
        previous_position = position;
        output_index = output_index + 1u;
      }
    }
  }

  // The final endpoint is emitted once, completing the last segment and
  // making the duration endpoint exact even when duration*FPS was fractional.
  let final_sample = raw_sample(seed, last);
  let final_position = final_sample.position.xyz;
  if (output_index > 0u) {
    arc_length = arc_length + distance(previous_position, final_position);
  }
  var final_velocity = vec3<f32>(0.0);
  if (params.sample_count > 1u) {
    let previous = raw_sample(seed, last - 1u);
    let p0 = control_before(seed, last - 1u);
    let p1 = previous.position.xyz;
    let p2 = final_position;
    let p3 = control_after(seed, last - 1u, last);
    final_velocity = catmull_rom_derivative(p0, p1, p2, p3, 1.0) /
      max(final_sample.metadata.x - previous.metadata.x, 0.000001);
  }
  write_sample(
    seed,
    output_index,
    final_position,
    final_velocity,
    final_sample.normal.xyz,
    final_sample.metadata.x,
    arc_length,
  );
  output_index = output_index + 1u;
  trajectory_counts[seed] = output_index;
}
