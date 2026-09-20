// One uniform block is shared by both shader stages.  The first 64 bytes are
// the column-major view-projection matrix; the next vec4 holds the camera;
// the final vec4 selects shader-side sRGB encoding for non-sRGB surfaces.
struct Uniforms { view_proj: mat4x4<f32>, camera: vec4<f32>, encode_srgb: vec4<f32> };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// The Rust Vertex layout supplies position at location 0 and normal at 1.
struct In { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32> };

// Animated vertices carry their four influences, but the matrices are kept in
// a storage palette so only the small palette changes for each animation
// frame. `base_transform` handles ordinary scene nodes and zero-weight
// vertices without requiring a second draw call.
struct SkinnedIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) base_transform: u32,
  @location(3) joints: vec4<u32>,
  @location(4) weights: vec4<f32>,
  @location(5) morph_base: u32,
  @location(6) morph_count: u32,
  @location(7) morph_weight_base: u32,
};

struct SkinTransform { matrix: mat4x4<f32>, normal: mat4x4<f32> };
@group(1) @binding(0) var<storage, read> transforms: array<SkinTransform>;
@group(1) @binding(1) var<storage, read> morph_positions: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read> morph_weights: array<f32>;

// The vertex stage forwards world-space values to the fragment stage for the
// simple per-fragment headlight calculation below.
struct Out { @builtin(position) position: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32> };

// Transform each mesh vertex into clip space.  Positions are already in world
// space because this small demo has no model transform.
@vertex fn vs_main(v: In) -> Out { var o:Out; o.position=uniforms.view_proj*vec4(v.position,1.0); o.world=v.position; o.normal=v.normal; return o; }

@vertex fn vs_skinned(v: SkinnedIn) -> Out {
  var local_position = v.position;
  for (var morph_target = 0u; morph_target < v.morph_count; morph_target = morph_target + 1u) {
    local_position = local_position +
      morph_positions[v.morph_base + morph_target].xyz *
      morph_weights[v.morph_weight_base + morph_target];
  }
  var position = vec4(0.0);
  var normal = vec4(0.0);
  var total = 0.0;
  for (var slot = 0u; slot < 4u; slot = slot + 1u) {
    let weight = v.weights[slot];
    if (weight > 0.0) {
      let transform = transforms[v.joints[slot]];
      position = position + transform.matrix * vec4(local_position, 1.0) * weight;
      normal = normal + transform.normal * vec4(v.normal, 0.0) * weight;
      total = total + weight;
    }
  }
  if (total <= 0.00001) {
    let transform = transforms[v.base_transform];
    position = transform.matrix * vec4(local_position, 1.0);
    normal = transform.normal * vec4(v.normal, 0.0);
  }
  var o:Out;
  o.position = uniforms.view_proj * position;
  o.world = position.xyz;
  o.normal = normalize(normal.xyz);
  return o;
}

// Use the camera as a light source.  The ambient term keeps back-facing or
// nearly unlit triangles visible against the dark render clear color.
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
  let low = color * 12.92;
  let high = 1.055 * pow(color, vec3(1.0 / 2.4)) - vec3(0.055);
  return select(low, high, color > vec3(0.0031308));
}

@fragment fn fs_main(v:Out) -> @location(0) vec4<f32> {
  let light=normalize(uniforms.camera.xyz-v.world);
  let diffuse=max(dot(normalize(v.normal),light),0.0);
  let linear=vec3(0.72,0.75,0.80)*(0.2+0.8*diffuse);
  let color=mix(linear,linear_to_srgb(linear),uniforms.encode_srgb.x);
  return vec4(color,1.0);
}
