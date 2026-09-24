// Camera-facing rings at the currently skinned positions of motion-line seeds.
// Six generated vertices form two triangles per seed; no geometry stage is
// needed in WebGPU. Vertex words use the packed SkinnedVertex layout.
struct Uniforms { view_proj: mat4x4<f32>, camera: vec4<f32>, encode_srgb: vec4<f32> };
struct SkinTransform { matrix: mat4x4<f32>, normal: mat4x4<f32> };
struct MotionLineStyle { timing: vec4<f32>, widths: vec4<f32>, viewport: vec4<f32> };

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> transforms: array<SkinTransform>;
@group(1) @binding(1) var<storage, read> morph_positions: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> morph_weights: array<f32>;
@group(2) @binding(0) var<storage, read> vertex_words: array<u32>;
@group(2) @binding(1) var<storage, read> seeds: array<u32>;
@group(2) @binding(2) var<uniform> style: MotionLineStyle;

fn word(vertex: u32, field: u32) -> u32 {
  return vertex_words[vertex * 18u + field];
}

fn skinned_position(vertex: u32) -> vec4<f32> {
  var local = vec3<f32>(
    bitcast<f32>(word(vertex, 0u)),
    bitcast<f32>(word(vertex, 1u)),
    bitcast<f32>(word(vertex, 2u)),
  );
  for (var morph_target = 0u; morph_target < word(vertex, 16u); morph_target = morph_target + 1u) {
    let offset = word(vertex, 15u) + morph_target;
    local += morph_positions[offset].xyz *
      morph_weights[word(vertex, 17u) + morph_target];
  }
  var position = vec4<f32>(0.0);
  var total = 0.0;
  for (var slot = 0u; slot < 4u; slot = slot + 1u) {
    let weight = bitcast<f32>(word(vertex, 11u + slot));
    if (weight > 0.0) {
      position += transforms[word(vertex, 7u + slot)].matrix * vec4<f32>(local, 1.0) * weight;
      total += weight;
    }
  }
  if (total <= 0.00001) {
    position = transforms[word(vertex, 6u)].matrix * vec4<f32>(local, 1.0);
  }
  return position;
}

struct RingOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex fn ring_vertex(
  @builtin(vertex_index) corner: u32,
  @builtin(instance_index) seed_index: u32,
) -> RingOut {
  // Two triangles in counterclockwise order in screen space.
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  );
  let uv = corners[corner];
  var clip = uniforms.view_proj * skinned_position(seeds[seed_index]);
  // Moving in NDC by a pixel-sized offset is a view-orthogonal billboard.
  // Its radius stays constant as the animated vertex moves in depth.
  let viewport = max(style.viewport.xy, vec2<f32>(1.0));
  let offset = uv * (2.0 * 10.0 / viewport) * clip.w;
  clip.x += offset.x;
  clip.y += offset.y;
  // A small depth lift keeps a seed on the surface from losing a depth tie.
  clip.z -= 1.0e-5 * clip.w;
  var out: RingOut;
  out.position = clip;
  out.uv = uv;
  return out;
}

struct RingFragmentOut {
  @location(0) accumulation: vec4<f32>,
  @location(1) revealage: vec4<f32>,
};

@fragment fn ring_fragment(input: RingOut) -> RingFragmentOut {
  let radius = length(input.uv);
  let aa = max(fwidth(radius), 0.01);
  let alpha = smoothstep(0.45 - aa, 0.45 + aa, radius) *
    (1.0 - smoothstep(0.83 - aa, 0.83 + aa, radius));
  if (alpha <= 0.0) { discard; }
  // Dark rings remain legible on the white mesh and white slide background.
  let color = vec3<f32>(0.035, 0.055, 0.09);
  var out: RingFragmentOut;
  out.accumulation = vec4<f32>(color * alpha * 8.0, alpha * 8.0);
  out.revealage = vec4<f32>(0.0, 0.0, 0.0, alpha);
  return out;
}
