// Uniform trace pass for one animated surface mesh.
//
// Every invocation owns one (seed, uniform-time-sample) pair. The seed is an
// actual vertex index, while the sample's pose comes from a palette prepared
// by the animation sampler. Position and the skinned surface normal are kept
// together so the later styling stage can use the normal at the time of the
// trajectory sample.

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

struct SkinTransform { matrix: mat4x4<f32>, normal: mat4x4<f32> };

struct TrajectorySample {
  position: vec4<f32>,
  velocity: vec4<f32>,
  normal: vec4<f32>,
  metadata: vec4<f32>, // x = time, y = arc length
};

@group(0) @binding(0) var<storage, read> vertex_words: array<u32>;
@group(0) @binding(1) var<storage, read> palettes: array<SkinTransform>;
@group(0) @binding(2) var<storage, read> morph_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> morph_weights: array<f32>;
@group(0) @binding(4) var<storage, read> seeds: array<u32>;
@group(0) @binding(5) var<storage, read_write> trajectories: array<TrajectorySample>;
@group(0) @binding(6) var<uniform> params: MotionLineParams;

fn word(vertex: u32, field: u32) -> u32 {
  return vertex_words[vertex * params.vertex_word_stride + field];
}

fn f32_word(vertex: u32, field: u32) -> f32 {
  return bitcast<f32>(word(vertex, field));
}

fn local_position(vertex_index: u32, sample_index: u32) -> vec3<f32> {
  var position = vec3<f32>(
    f32_word(vertex_index, 0u),
    f32_word(vertex_index, 1u),
    f32_word(vertex_index, 2u),
  );
  let morph_base = word(vertex_index, 15u);
  let morph_count = word(vertex_index, 16u);
  let morph_weight_base = word(vertex_index, 17u);
  for (var morph_target = 0u; morph_target < morph_count; morph_target = morph_target + 1u) {
    position = position + morph_positions[morph_base + morph_target].xyz *
      morph_weights[sample_index * params.morph_weight_stride + morph_weight_base + morph_target];
  }
  return position;
}

fn transform_position(vertex_index: u32, sample_index: u32) -> vec3<f32> {
  let position = local_position(vertex_index, sample_index);
  let palette_base = sample_index * params.palette_stride;
  let base_transform = word(vertex_index, 6u);
  var transformed = vec4<f32>(0.0);
  var total = 0.0;
  for (var slot = 0u; slot < 4u; slot = slot + 1u) {
    let weight = f32_word(vertex_index, 11u + slot);
    if (weight > 0.0) {
      let joint = word(vertex_index, 7u + slot);
      transformed = transformed +
        palettes[palette_base + joint].matrix * vec4<f32>(position, 1.0) * weight;
      total = total + weight;
    }
  }
  if (total <= 0.00001) {
    transformed = palettes[palette_base + base_transform].matrix * vec4<f32>(position, 1.0);
  }
  return transformed.xyz;
}

fn transform_normal(vertex_index: u32, sample_index: u32) -> vec3<f32> {
  // The compact animated vertex format stores authored normals but no normal
  // morph deltas. Thus skeletal/node normal skinning is exact here; a future
  // morph-normal stream can be added without changing the trajectory ABI.
  let normal = vec3<f32>(
    f32_word(vertex_index, 3u),
    f32_word(vertex_index, 4u),
    f32_word(vertex_index, 5u),
  );
  let palette_base = sample_index * params.palette_stride;
  let base_transform = word(vertex_index, 6u);
  var transformed = vec4<f32>(0.0);
  var total = 0.0;
  for (var slot = 0u; slot < 4u; slot = slot + 1u) {
    let weight = f32_word(vertex_index, 11u + slot);
    if (weight > 0.0) {
      let joint = word(vertex_index, 7u + slot);
      transformed = transformed +
        palettes[palette_base + joint].normal * vec4<f32>(normal, 0.0) * weight;
      total = total + weight;
    }
  }
  if (total <= 0.00001) {
    transformed = palettes[palette_base + base_transform].normal * vec4<f32>(normal, 0.0);
  }
  let length_squared = dot(transformed.xyz, transformed.xyz);
  if (length_squared <= 0.00000001) {
    return vec3<f32>(0.0, 1.0, 0.0);
  }
  return normalize(transformed.xyz);
}

fn sample_time(sample_index: u32) -> f32 {
  if (sample_index + 1u >= params.sample_count) {
    return params.duration;
  }
  return f32(sample_index) / params.samples_per_second;
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let seed_index = invocation.x;
  let sample_index = invocation.y;
  if (seed_index >= params.seed_count || sample_index >= params.sample_count) {
    return;
  }
  let vertex_index = seeds[seed_index];
  let output_index = seed_index * params.sample_count + sample_index;
  trajectories[output_index] = TrajectorySample(
    vec4<f32>(transform_position(vertex_index, sample_index), 1.0),
    vec4<f32>(0.0),
    vec4<f32>(transform_normal(vertex_index, sample_index), 0.0),
    vec4<f32>(sample_time(sample_index), 0.0, 0.0, 0.0),
  );
}
