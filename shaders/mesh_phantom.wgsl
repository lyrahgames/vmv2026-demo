struct PhantomOpacity { value: vec4<f32> };
@group(2) @binding(0) var<uniform> phantom_opacity: PhantomOpacity;

@fragment fn fs_phantom(v: Out) -> @location(0) vec4<f32> {
  let normal = normalize(v.normal);
  let view_direction = normalize(uniforms.camera.xyz - v.world);
  let view_alignment = abs(dot(normal, view_direction));
  let light = toon_light(view_alignment);
  let surface_linear = vec3(light);
  let contour = silhouette_mask(view_alignment);
  let contour_linear = vec3(0.08, 0.09, 0.11);
  let linear = mix(surface_linear, contour_linear, contour);
  let color = mix(linear, linear_to_srgb(linear), uniforms.encode_srgb.x);
  return vec4(color, clamp(phantom_opacity.value.x, 0.0, 1.0));
}
