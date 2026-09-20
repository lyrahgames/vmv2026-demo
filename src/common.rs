//! Small shared prelude for types used throughout the renderer.
//!
//! Keep this limited to genuinely cross-cutting types.  Feature-specific APIs
//! such as wgpu, winit, Lua, and web-sys stay imported in the modules that use
//! them, so each module's platform dependencies remain obvious.

// Error types and macros used by parsing, native startup, and GPU setup.
pub use anyhow::{Context, Result, anyhow, bail};

// Math primitives shared by mesh loading, cameras, input, and the web bridge.
pub use glam::{Mat4, Quat, Vec2, Vec3};

// Shared ownership and interior mutability used by the event-loop bridges.
pub use futures::channel::oneshot;
pub use std::{
  cell::{Cell, RefCell},
  rc::Rc,
  sync::Arc,
};
