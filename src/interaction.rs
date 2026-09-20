//! Minimal mouse controls for orbiting, panning, and zooming the camera.

use crate::{camera::Camera, common::*};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};

#[derive(Default)]
pub struct Interaction {
  // The active mouse button determines which camera operation a drag means.
  button: Option<MouseButton>,
  // The previous cursor position is needed to turn absolute cursor events
  // into a relative drag delta.
  last:   Option<Vec2>,
}
impl Interaction {
  /// Converts winit window events into camera mutations.
  pub fn event(&mut self, event: &WindowEvent, camera: &mut Camera) {
    match event {
      WindowEvent::MouseInput {
        state: ElementState::Pressed,
        button,
        ..
      } => self.button = Some(*button),
      WindowEvent::MouseInput {
        state: ElementState::Released,
        ..
      } => {
        // A release ends both the operation and the delta chain.  The
        // next press therefore starts without a jump.
        self.button = None;
        self.last = None;
      }
      WindowEvent::CursorMoved { position, .. } => {
        let p = Vec2::new(position.x as f32, position.y as f32);
        if let Some(last) = self.last {
          // Browser/window coordinates grow downward; camera coordinates grow upward.
          let delta = p - last;
          let d = Vec2::new(delta.x, -delta.y) * 0.01;
          if self.button == Some(MouseButton::Left) {
            // Left drag rotates around the target.
            orbit(camera, d);
          } else if self.button == Some(MouseButton::Right) {
            // Right drag translates both eye and target together.
            pan(camera, d);
          }
        }
        self.last = Some(p);
      }
      WindowEvent::MouseWheel { delta, .. } => {
        // Winit reports either logical wheel lines or physical pixels;
        // normalize both into a small logarithmic zoom step.
        let y = match delta {
          MouseScrollDelta::LineDelta(_, y) => *y,
          MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
        };
        let factor = (-y * 0.12).exp();
        let v = camera.eye - camera.target;
        camera.eye = camera.target + v * factor.max(0.05);
        camera.update_planes(v.length());
      }
      _ => {}
    }
  }
}

/// Rotates the eye around the target while keeping the orbit radius constant.
fn orbit(c: &mut Camera, d: Vec2) {
  let v = c.eye - c.target;
  let r = v.length().max(0.001);
  let yaw = d.x;
  let pitch = d.y;
  // First yaw around the current up axis, then pitch around the camera's
  // local horizontal axis.  Quaternions avoid accumulating Euler-angle drift.
  let q = Quat::from_axis_angle(c.up.normalize_or_zero(), -yaw)
    * Quat::from_axis_angle((c.eye - c.target).cross(c.up).normalize_or_zero(), -pitch);
  let nv = (q * v).normalize() * r.max(0.02);
  c.eye = c.target + nv;
}

/// Moves the eye and target together in the camera's image plane.
fn pan(c: &mut Camera, d: Vec2) {
  let view = (c.target - c.eye).normalize_or_zero();
  let right = view.cross(c.up).normalize_or_zero();
  let up = right.cross(view).normalize_or_zero();
  // Scale panning by distance so the same mouse motion feels similar at
  // different zoom levels.
  let amount = (c.eye - c.target).length() * 0.8;
  let shift = (-right * d.x + up * d.y) * amount;
  c.eye += shift;
  c.target += shift;
}
