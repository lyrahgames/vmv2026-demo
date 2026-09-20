// Low-rate position-only trace used by spacetime seed selection.
//
// This pass intentionally omits normals, velocities, arc lengths, and all
// other styling data. Its output is consumed immediately by the selection
// pass and then destroyed, keeping the all-vertex spacetime probe cheap.

struct SeedTraceParams {
  vertex_count:        u32,
  sample_count:        u32,
  vertex_word_stride:  u32,
  palette_stride:      u32,
  morph_weight_stride: u32,
  samples_per_second:  f32,
  duration:            f32,
  padding0:            u32,
};

struct SkinTransform { matrix: mat4x4<f32>, normal: mat4x4<f32> };

@group(0) @binding(0) var<storage, read> vertex_words: array<u32>;
@group(0) @binding(1) var<storage, read> palettes: array<SkinTransform>;
@group(0) @binding(2) var<storage, read> morph_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> morph_weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: SeedTraceParams;

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

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let vertex_index = invocation.x;
  let sample_index = invocation.y;
  if (vertex_index >= params.vertex_count || sample_index >= params.sample_count) {
    return;
  }
  // Time-major storage makes each selection-time slice contiguous, which is
  // the access pattern of the max-min selection pass.
  positions[sample_index * params.vertex_count + vertex_index] =
    vec4<f32>(transform_position(vertex_index, sample_index), 1.0);
}
