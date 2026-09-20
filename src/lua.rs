//! Native Lua adapter for the readiness-aware viewer task queue.
//!
//! Lua runs before winit's event loop. Its functions enqueue work instead of
//! touching GPU state. `load_obj` parses synchronously for useful startup
//! errors, but the resulting mesh is still applied only after viewer creation.

use crate::{
  application::TaskQueue,
  camera::CameraConfig,
  common::*,
  mesh::Mesh,
  motion_lines::{MotionLineConfig, SeedSelectionAlgorithm},
  scene::AnimatedScene,
  scene::AnimationInfo,
};
use mlua::{Lua, Table};

fn vec3(t: Table) -> mlua::Result<Vec3> {
  // Lua arrays are one-based, unlike Rust slices and JavaScript arrays.
  Ok(Vec3::new(t.get(1)?, t.get(2)?, t.get(3)?))
}

fn color3(t: Table) -> mlua::Result<[f32; 3]> {
  // Colors use the same one-based normalized RGB array convention as camera
  // vectors: set_background_color({ red, green, blue }). The viewer clamps
  // the values when the queued action is applied.
  Ok([t.get(1)?, t.get(2)?, t.get(3)?])
}

/// Converts the format-neutral animation metadata into the Lua table shape
/// shared by `load_gltf` and `load_fbx`. Keeping this adapter shared prevents
/// the two loaders from slowly developing different scripting APIs.
fn animation_infos_table(lua: &Lua, infos: Vec<AnimationInfo>) -> mlua::Result<Table> {
  let result = lua.create_table()?;
  for (animation_index, info) in infos.into_iter().enumerate() {
    let animation_table = lua.create_table()?;
    animation_table.set("name", info.name)?;
    animation_table.set("duration", info.duration)?;
    let channels = lua.create_table()?;
    for (channel_index, channel) in info.channels.into_iter().enumerate() {
      let channel_table = lua.create_table()?;
      channel_table.set("node", channel.node)?;
      channel_table.set("property", channel.property)?;
      channel_table.set("interpolation", channel.interpolation)?;
      channels.set(channel_index + 1, channel_table)?;
    }
    animation_table.set("channels", channels)?;
    result.set(animation_index + 1, animation_table)?;
  }
  Ok(result)
}

/// Executes a Lua script and returns tasks for [`crate::application::App`].
pub fn tasks(path: &str) -> Result<TaskQueue> {
  let lua = Lua::new();
  let queue = TaskQueue::new();

  let load = queue.clone();
  let load_obj = lua
    .create_function(move |_, path: String| {
      let mesh = Mesh::from_file(&path).map_err(|e| mlua::Error::external(e.to_string()))?;
      load.set_mesh(mesh);
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("load_obj", load_obj)
    .map_err(|e| anyhow!(e.to_string()))?;

  // glTF parsing happens before the event loop, just like OBJ parsing. The
  // returned Lua table makes every animation and every channel inspectable,
  // while the scene itself is deferred until a GPU-backed Viewer exists.
  let load = queue.clone();
  let load_gltf = lua
    .create_function(move |lua, path: String| {
      let scene =
        AnimatedScene::from_file(&path).map_err(|e| mlua::Error::external(e.to_string()))?;
      let infos = scene.animations();
      load.set_scene(scene);
      animation_infos_table(lua, infos)
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("load_gltf", load_gltf)
    .map_err(|e| anyhow!(e.to_string()))?;

  // FBX uses the exact same script contract as glTF. The decoder runs
  // before winit, while `set_scene` queues the GPU-dependent replacement for
  // the moment the viewer has actually been constructed.
  let load = queue.clone();
  let load_fbx = lua
    .create_function(move |lua, path: String| {
      let scene =
        AnimatedScene::from_file(&path).map_err(|e| mlua::Error::external(e.to_string()))?;
      let infos = scene.animations();
      load.set_scene(scene);
      animation_infos_table(lua, infos)
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("load_fbx", load_fbx)
    .map_err(|e| anyhow!(e.to_string()))?;

  let reset = queue.clone();
  let reset_camera = lua
    .create_function(move |_, ()| {
      reset.reset_camera();
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("reset_camera", reset_camera)
    .map_err(|e| anyhow!(e.to_string()))?;

  let set = queue.clone();
  let set_camera = lua
    .create_function(move |_, t: Table| {
      set.set_camera(CameraConfig {
        eye:    vec3(t.get("eye")?)?,
        target: vec3(t.get("target")?)?,
        up:     vec3(t.get("up")?)?,
        fov:    t.get("fov")?,
      });
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("set_camera", set_camera)
    .map_err(|e| anyhow!(e.to_string()))?;

  let background = queue.clone();
  let set_background_color = lua
    .create_function(move |_, color: Table| {
      background.set_background_color(color3(color)?);
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("set_background_color", set_background_color)
    .map_err(|e| anyhow!(e.to_string()))?;

  let select = queue.clone();
  let select_animation = lua
    .create_function(move |_, index: usize| {
      // Lua uses one-based indexes, while glTF and Rust use zero-based
      // indexes. Rejecting zero catches a common script mistake early.
      if index == 0 {
        return Err(mlua::Error::external("animation indexes start at 1"));
      }
      select.select_animation(index - 1);
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("select_animation", select_animation)
    .map_err(|e| anyhow!(e.to_string()))?;

  let play = queue.clone();
  lua
    .globals()
    .set(
      "play_animation",
      lua
        .create_function(move |_, ()| {
          play.play_animation();
          Ok(())
        })
        .map_err(|e| anyhow!(e.to_string()))?,
    )
    .map_err(|e| anyhow!(e.to_string()))?;

  let pause = queue.clone();
  lua
    .globals()
    .set(
      "pause_animation",
      lua
        .create_function(move |_, ()| {
          pause.pause_animation();
          Ok(())
        })
        .map_err(|e| anyhow!(e.to_string()))?,
    )
    .map_err(|e| anyhow!(e.to_string()))?;

  let time = queue.clone();
  lua
    .globals()
    .set(
      "set_animation_time",
      lua
        .create_function(move |_, value: f32| {
          time.set_animation_time(value);
          Ok(())
        })
        .map_err(|e| anyhow!(e.to_string()))?,
    )
    .map_err(|e| anyhow!(e.to_string()))?;

  let speed = queue.clone();
  lua
    .globals()
    .set(
      "set_animation_speed",
      lua
        .create_function(move |_, value: f32| {
          speed.set_animation_speed(value);
          Ok(())
        })
        .map_err(|e| anyhow!(e.to_string()))?,
    )
    .map_err(|e| anyhow!(e.to_string()))?;

  let all_motion_lines = queue.clone();
  let set_motion_lines_all = lua
    .create_function(move |_, fps: f32| {
      all_motion_lines.configure_motion_lines(MotionLineConfig {
        seed_selection:    SeedSelectionAlgorithm::AllVertices,
        frames_per_second: fps,
      });
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("set_motion_lines_all", set_motion_lines_all)
    .map_err(|e| anyhow!(e.to_string()))?;

  let random_motion_lines = queue.clone();
  let set_motion_lines_random = lua
    .create_function(move |_, (count, fps): (usize, f32)| {
      random_motion_lines.configure_motion_lines(MotionLineConfig {
        seed_selection:    SeedSelectionAlgorithm::RandomVertices { count },
        frames_per_second: fps,
      });
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("set_motion_lines_random", set_motion_lines_random)
    .map_err(|e| anyhow!(e.to_string()))?;

  let uniform_motion_lines = queue.clone();
  let set_motion_lines_uniform = lua
    .create_function(move |_, (count, fps): (usize, f32)| {
      uniform_motion_lines.configure_motion_lines(MotionLineConfig {
        seed_selection:    SeedSelectionAlgorithm::UniformVertices { count },
        frames_per_second: fps,
      });
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("set_motion_lines_uniform", set_motion_lines_uniform)
    .map_err(|e| anyhow!(e.to_string()))?;

  let spacetime_motion_lines = queue.clone();
  let set_motion_lines_uniform_spacetime = lua
    .create_function(move |_, (count, sampling_rate, fps): (usize, f32, f32)| {
      spacetime_motion_lines.configure_motion_lines(MotionLineConfig {
        seed_selection:    SeedSelectionAlgorithm::UniformSpacetimeVertices {
          count,
          sampling_rate,
        },
        frames_per_second: fps,
      });
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set(
      "set_motion_lines_uniform_spacetime",
      set_motion_lines_uniform_spacetime,
    )
    .map_err(|e| anyhow!(e.to_string()))?;

  let clear_motion_lines = queue.clone();
  let clear_motion_lines_function = lua
    .create_function(move |_, ()| {
      clear_motion_lines.clear_motion_lines();
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("clear_motion_lines", clear_motion_lines_function)
    .map_err(|e| anyhow!(e.to_string()))?;

  let print_memory = queue.clone();
  let print_memory_usage = lua
    .create_function(move |_, label: Option<String>| {
      print_memory.print_memory_usage(label.unwrap_or_else(|| "lua checkpoint".to_owned()));
      Ok(())
    })
    .map_err(|e| anyhow!(e.to_string()))?;
  lua
    .globals()
    .set("print_memory_usage", print_memory_usage)
    .map_err(|e| anyhow!(e.to_string()))?;

  lua
    .load(std::fs::read_to_string(path)?)
    .set_name(path)
    .exec()
    .map_err(|e| anyhow!(e.to_string()))?;
  Ok(queue)
}
