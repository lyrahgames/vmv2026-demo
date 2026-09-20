//! Camera state and projection math shared by native and web viewers.

use crate::common::*;

/// A complete camera pose supplied by a script or by the JavaScript API.
#[derive(Clone, Debug)]
pub struct CameraConfig {
  /// World-space position of the camera.
  pub eye:    Vec3,
  /// World-space point the camera looks at.
  pub target: Vec3,
  /// Approximate world-space up direction used to remove roll.
  pub up:     Vec3,
  /// Vertical field of view in degrees.
  pub fov:    f32,
}

/// Camera pose used when the caller wants the viewer to supply the look-at
/// point from the mesh currently displayed by the renderer.
#[derive(Clone, Debug)]
pub struct CameraFollowConfig {
  /// Camera position relative to the current mesh center.
  pub offset: Vec3,
  /// Approximate world-space up direction used to remove roll.
  pub up:     Vec3,
  /// Vertical field of view in degrees.
  pub fov:    f32,
}

/// Mutable camera state used to build the vertex shader's view-projection.
#[derive(Clone, Debug)]
pub struct Camera {
  /// Current world-space camera position.
  pub eye:          Vec3,
  /// Current look-at point.
  pub target:       Vec3,
  /// Normalized up direction.
  pub up:           Vec3,
  /// Vertical field of view, in degrees.
  pub vertical_fov: f32,
  /// Width divided by height of the render surface.
  pub aspect:       f32,
  /// Near clipping plane.  Objects closer than this are discarded.
  pub near:         f32,
  /// Far clipping plane.  Objects farther than this are discarded.
  pub far:          f32,
}

impl Camera {
  /// Creates a useful default camera for a surface with the given aspect.
  pub fn new(aspect: f32) -> Self {
    Self {
      eye: Vec3::new(2.5, 1.8, 3.0),
      target: Vec3::ZERO,
      up: Vec3::Y,
      vertical_fov: 45.0,
      aspect,
      near: 0.01,
      far: 100.0,
    }
  }

  /// Replaces the pose while sanitizing values that could break projection.
  /// The caller updates the clipping planes with the bounds of the mesh it is
  /// rendering; the camera itself does not know that mesh radius.
  ///
  /// A zero-length up vector cannot define a camera orientation, so it falls
  /// back to world up.  The FOV is clamped because extreme values make the
  /// perspective matrix unstable or visually unusable.
  pub fn set_camera(&mut self, config: CameraConfig) {
    self.eye = config.eye;
    self.target = config.target;
    self.up = if config.up.length_squared() > 0.0 {
      config.up.normalize()
    } else {
      Vec3::Y
    };
    self.vertical_fov = config.fov.clamp(1.0, 179.0);
  }

  /// Builds the matrix that transforms world vertices into clip space.
  pub fn view_projection(&self) -> Mat4 {
    let view = Mat4::look_at_rh(self.eye, self.target, self.up);
    let projection = Mat4::perspective_rh(
      self.vertical_fov.to_radians(),
      self.aspect.max(0.01),
      self.near,
      self.far,
    );
    projection * view
  }

  /// Derives clipping planes from the current camera distance and mesh size.
  ///
  /// Keeping the far plane near the scene improves depth precision, while the
  /// minimum gap prevents the near and far planes from collapsing together.
  pub fn update_planes(&mut self, radius: f32) {
    let distance = (self.eye - self.target).length().max(radius * 0.1);
    self.near = (radius * 0.01).max(0.001).min(distance * 0.25);
    self.far = (distance + radius * 8.0).max(self.near + 1.0);
  }

  /// Frames an axis-aligned mesh bounding box without knowing its topology.
  ///
  /// The bounding-box diagonal supplies a conservative sphere radius.  The
  /// larger of the vertical and horizontal FOV distances is used so the mesh
  /// remains visible even in a wide or unusually narrow canvas.
  pub fn reset_for_bounds(&mut self, min: Vec3, max: Vec3) {
    let center = (min + max) * 0.5;
    // A sphere around the box is conservative but works for every mesh.
    let radius = ((max - min) * 0.5).length().max(0.01);
    let half_fov = (self.vertical_fov.to_radians() * 0.5).tan();
    let vertical_distance = radius / half_fov;
    let horizontal_distance = radius / (half_fov * self.aspect.max(0.01));
    let distance = vertical_distance.max(horizontal_distance) * 1.35;
    // Preserve the previous viewing direction so reset does not abruptly
    // rotate the user to a completely different side of the object.
    let direction = (self.eye - self.target)
      .try_normalize()
      .unwrap_or(Vec3::new(1.0, 0.65, 1.0).normalize());
    self.target = center;
    self.eye = center + direction * distance.max(radius * 2.0);
    self.update_planes(radius);
  }
}
