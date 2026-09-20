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
// view-dependent toon and silhouette calculation below.
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

// This is the stepped view-lighting curve used by the teaser surface shader.
// The view direction is used instead of a fixed world-space light so the
// result remains stable when the camera is orbited around the model.
fn toon_light(view_alignment: f32) -> f32 {
  let light = pow(clamp(view_alignment, 0.0, 1.0), 0.5);
  if (light <= 0.55) {
    return 0.4;
  }
  if (light <= 0.8) {
    return 0.8;
  }
  return 1.0;
}

// A smooth normal-based contour darkens the grazing-angle band of the
// surface. This is the silhouette analogue of the teaser's optional
// edge-distance wireframe: it needs no extra barycentric vertex data and
// works for both static and GPU-skinned meshes. `fwidth` keeps the narrow band
// from flickering when its projected width is close to one pixel.
fn silhouette_mask(view_alignment: f32) -> f32 {
  let anti_alias = max(fwidth(view_alignment) * 1.5, 0.005);
  // Extend the transition farther toward front-facing normals so the contour
  // reads as a broad graphic stroke rather than a narrow one-pixel rim.
  return 1.0 - smoothstep(0.035 - anti_alias, 0.28 + anti_alias, view_alignment);
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
  let low = color * 12.92;
  let high = 1.055 * pow(color, vec3(1.0 / 2.4)) - vec3(0.055);
  return select(low, high, color > vec3(0.0031308));
}

@fragment fn fs_main(v:Out) -> @location(0) vec4<f32> {
  let normal = normalize(v.normal);
  let view_direction = normalize(uniforms.camera.xyz - v.world);
  // The teaser uses abs(view-space-normal.z), which is equivalent to the
  // absolute view-normal alignment here and keeps the reverse side readable.
  let view_alignment = abs(dot(normal, view_direction));
  let light = toon_light(view_alignment);
  // The teaser's surface shader uses a neutral grayscale material. Keeping
  // the brightest toon band at one makes front-facing regions pure white
  // instead of tinting them blue-gray.
  let surface_linear = vec3(light);

  // Silhouette pixels are deliberately dark rather than transparent. This
  // keeps the contour visible on the white scripted background while still
  // allowing the mesh depth pass to occlude motion lines correctly.
  let contour = silhouette_mask(view_alignment);
  let contour_linear = vec3(0.08, 0.09, 0.11);
  let linear = mix(surface_linear, contour_linear, contour);
  let color = mix(linear, linear_to_srgb(linear), uniforms.encode_srgb.x);
  return vec4(color,1.0);
}
