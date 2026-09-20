//! Shared library for the native and WebAssembly surface-mesh viewer.
//!
//! The same camera, mesh, glTF scene, interaction, application, and renderer modules are
//! compiled for both targets.  Only the outer control surfaces differ: native
//! uses Lua, while WASM exposes a JavaScript handle.

pub mod application;
pub mod camera;
pub mod common;
pub mod interaction;
pub mod mesh;
pub mod motion_lines;
pub mod scene;
pub mod viewer;

#[cfg(not(target_arch = "wasm32"))]
pub mod lua;
#[cfg(target_arch = "wasm32")]
pub mod web;
