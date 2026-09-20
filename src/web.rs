//! JavaScript/WebAssembly bridge for the readiness-aware task queue.
//!
//! JavaScript methods enqueue work immediately, even before the GPU is ready.
//! `ready` and `load_obj_text_async` return JavaScript promises for callers
//! that need to wait until queued graphics work has actually been processed.

use crate::{
  application::{App, ApplicationEvent, TaskQueue},
  camera::{CameraConfig, CameraFollowConfig},
  common::*,
  mesh::Mesh,
  motion_lines::{MotionLineConfig, SeedSelectionAlgorithm},
  scene::{AnimatedScene, AnimationInfo},
};
use js_sys::Uint8Array;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;
use winit::platform::web::EventLoopExtWebSys;

struct Runtime {
  proxy:           winit::event_loop::EventLoopProxy<ApplicationEvent>,
  queue:           TaskQueue,
  // This mirror is updated before Attach reaches winit. It rejects a late
  // JavaScript callback immediately, before it can replace the queued mesh
  // seed for the newly visible slide.
  active_id:       Rc<Cell<u64>>,
  // Independent handles are allowed to enqueue work even though they are
  // not the ordinary slide singleton currently shown by navigation.
  independent_ids: Rc<RefCell<std::collections::HashSet<u64>>>,
}

thread_local! {
    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
    static NEXT_ID: Cell<u64> = const { Cell::new(1) };
}

#[wasm_bindgen]
pub struct ViewerHandle {
  id:          u64,
  independent: bool,
}

#[wasm_bindgen]
impl ViewerHandle {
  /// Attaches a canvas to the one page-wide winit event loop.
  #[wasm_bindgen(constructor)]
  pub fn new(canvas: HtmlCanvasElement, independent: bool) -> Result<ViewerHandle, JsValue> {
    let id = NEXT_ID.with(|next| {
      let id = next.get();
      next.set(id + 1);
      id
    });
    let existing = RUNTIME.with(|runtime| runtime.borrow().as_ref().map(|r| r.proxy.clone()));
    if let Some(proxy) = existing {
      RUNTIME.with(|runtime| {
        let runtime = runtime.borrow();
        let runtime = runtime
          .as_ref()
          .expect("runtime disappeared while creating a handle");
        if independent {
          runtime.independent_ids.borrow_mut().insert(id);
        } else {
          runtime.active_id.set(id);
        }
      });
      let _ = proxy.send_event(ApplicationEvent::Attach {
        id,
        canvas,
        independent,
      });
    } else {
      let event_loop = winit::event_loop::EventLoop::<ApplicationEvent>::with_user_event()
        .build()
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
      let queue = TaskQueue::new();
      let proxy = event_loop.create_proxy();
      let app = App::new(queue.clone());
      let mut app = app;
      app.bind_queue(&event_loop);
      // Start one retained browser RAF dispatcher for the whole WASM
      // module. It renders every live canvas and avoids winit's fragile
      // per-canvas callback lifetime during Slidev transitions.
      crate::application::start_web_frame_loop(proxy.clone());
      // Register the first canvas before winit starts.  Sending Attach
      // immediately after spawn_app can race the initial resumed event,
      // leaving the event loop with no canvas to create a window for.
      app.set_initial_web_canvas(id, canvas, independent);
      let independent_ids = Rc::new(RefCell::new(std::collections::HashSet::new()));
      if independent {
        independent_ids.borrow_mut().insert(id);
      }
      RUNTIME.with(|runtime| {
        *runtime.borrow_mut() = Some(Runtime {
          proxy: proxy.clone(),
          queue,
          active_id: Rc::new(Cell::new(id)),
          independent_ids,
        })
      });
      event_loop.spawn_app(app);
    }
    Ok(Self { id, independent })
  }

  fn queue(&self) -> Option<TaskQueue> {
    RUNTIME.with(|runtime| {
      runtime.borrow().as_ref().and_then(|runtime| {
        (self.independent && runtime.independent_ids.borrow().contains(&self.id)
          || !self.independent && runtime.active_id.get() == self.id)
          .then(|| runtime.queue.clone())
      })
    })
  }

  /// Parses OBJ text and queues its replacement. This is fire-and-forget.
  pub fn load_obj_text(&self, text: String) -> Result<(), JsValue> {
    let mesh = Mesh::from_obj_text(&text).map_err(|e| JsValue::from_str(&e.to_string()))?;
    if let Some(queue) = self.queue() {
      queue.set_web_mesh(self.id, mesh);
    }
    Ok(())
  }

  /// Parses and applies an OBJ, resolving only when its viewer action ran.
  pub async fn load_obj_text_async(&self, text: String) -> Result<(), JsValue> {
    let mesh = Mesh::from_obj_text(&text).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let Some(queue) = self.queue() else {
      return Err(JsValue::from_str("viewer runtime has stopped"));
    };
    queue.set_web_mesh(self.id, mesh);
    self.ready().await
  }

  /// Parses a GLB or an embedded-buffer glTF supplied by JavaScript.
  ///
  /// Returning animation metadata after `ready()` gives callers one promise
  /// that covers parsing, GPU replacement, and viewer readiness.
  #[wasm_bindgen(js_name = loadGltfBytes)]
  pub async fn load_gltf_bytes(&self, bytes: Uint8Array) -> Result<JsValue, JsValue> {
    let scene = AnimatedScene::from_bytes(&bytes.to_vec())
      .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let infos = animation_infos_to_js(&scene.animations())?;
    let Some(queue) = self.queue() else {
      return Err(JsValue::from_str("viewer runtime has stopped"));
    };
    queue.set_web_scene(self.id, scene);
    self.ready().await?;
    Ok(infos)
  }

  /// Parses and applies a complete FBX file supplied by JavaScript.
  ///
  /// Unlike JSON glTF, binary FBX keeps its geometry and animation data in
  /// one file, so the browser only needs to fetch the file and pass its raw
  /// bytes to WASM. The returned metadata has the same shape and timing
  /// guarantees as `loadGltfBytes`.
  #[wasm_bindgen(js_name = loadFbxBytes)]
  pub async fn load_fbx_bytes(&self, bytes: Uint8Array) -> Result<JsValue, JsValue> {
    let scene = AnimatedScene::from_bytes(&bytes.to_vec())
      .map_err(|error| JsValue::from_str(&error.to_string()))?;
    let infos = animation_infos_to_js(&scene.animations())?;
    let Some(queue) = self.queue() else {
      return Err(JsValue::from_str("viewer runtime has stopped"));
    };
    queue.set_web_scene(self.id, scene);
    self.ready().await?;
    Ok(infos)
  }

  pub fn reset_camera(&self) {
    if let Some(queue) = self.queue() {
      queue.enqueue_web(self.id, |viewer| viewer.reset_camera());
    }
  }

  /// Sets the normalized linear-RGB clear color used by the active canvas.
  /// JavaScript passes `[red, green, blue]`; the Rust viewer clamps values so
  /// scripts can safely use ordinary normalized color constants.
  #[wasm_bindgen(js_name = setBackgroundColor)]
  pub fn set_background_color(&self, color: js_sys::Array) -> Result<(), JsValue> {
    let mut rgb = [0.0_f32; 3];
    for (index, channel) in rgb.iter_mut().enumerate() {
      let value = color
        .get(index as u32)
        .as_f64()
        .ok_or_else(|| JsValue::from_str("background color must be [red, green, blue]"))?;
      if !value.is_finite() {
        return Err(JsValue::from_str(
          "background color channels must be finite",
        ));
      }
      *channel = value as f32;
    }
    if let Some(queue) = self.queue() {
      queue.set_web_background_color(self.id, rgb);
    }
    Ok(())
  }

  pub fn resize(&self, width: u32, height: u32) {
    if let Some(queue) = self.queue() {
      let size = winit::dpi::PhysicalSize::new(width.max(1), height.max(1));
      queue.enqueue_web(self.id, move |viewer| viewer.resize(size));
    }
  }

  /// Selects a loaded glTF animation without starting it.
  #[wasm_bindgen(js_name = selectAnimation)]
  pub fn select_animation(&self, index: usize) {
    if let Some(queue) = self.queue() {
      queue.select_web_animation(self.id, index);
    }
  }

  /// Starts the selected glTF animation.
  #[wasm_bindgen(js_name = playAnimation)]
  pub fn play_animation(&self) {
    if let Some(queue) = self.queue() {
      queue.play_web_animation(self.id);
    }
  }

  /// Pauses playback at the current frame.
  #[wasm_bindgen(js_name = pauseAnimation)]
  pub fn pause_animation(&self) {
    if let Some(queue) = self.queue() {
      queue.pause_web_animation(self.id);
    }
  }

  /// Sets the selected animation time in seconds.
  #[wasm_bindgen(js_name = setAnimationTime)]
  pub fn set_animation_time(&self, time: f32) {
    if let Some(queue) = self.queue() {
      queue.set_web_animation_time(self.id, time);
    }
  }

  /// Sets the selected animation's playback multiplier.
  #[wasm_bindgen(js_name = setAnimationSpeed)]
  pub fn set_animation_speed(&self, speed: f32) {
    if let Some(queue) = self.queue() {
      queue.set_web_animation_speed(self.id, speed);
    }
  }

  /// Extracts and renders a trajectory for every source surface vertex.
  #[wasm_bindgen(js_name = setMotionLinesAll)]
  pub async fn set_motion_lines_all(&self, fps: f32) -> Result<(), JsValue> {
    self
      .set_motion_lines(MotionLineConfig {
        seed_selection:    SeedSelectionAlgorithm::AllVertices,
        frames_per_second: fps,
      })
      .await
  }

  /// Extracts and renders trajectories for `count` distinct source vertices.
  #[wasm_bindgen(js_name = setMotionLinesRandom)]
  pub async fn set_motion_lines_random(&self, count: usize, fps: f32) -> Result<(), JsValue> {
    self
      .set_motion_lines(MotionLineConfig {
        seed_selection:    SeedSelectionAlgorithm::RandomVertices { count },
        frames_per_second: fps,
      })
      .await
  }

  /// Extracts trajectories from a spatially uniform greedy vertex set. The
  /// selector starts at the farthest pair and repeatedly adds the candidate
  /// farthest from its closest existing seed.
  #[wasm_bindgen(js_name = setMotionLinesUniform)]
  pub async fn set_motion_lines_uniform(&self, count: usize, fps: f32) -> Result<(), JsValue> {
    self
      .set_motion_lines(MotionLineConfig {
        seed_selection:    SeedSelectionAlgorithm::UniformVertices { count },
        frames_per_second: fps,
      })
      .await
  }

  /// Selects seeds uniformly over a low-rate sample of the complete
  /// animation, then traces the selected seeds at the final line FPS.
  #[wasm_bindgen(js_name = setMotionLinesUniformSpacetime)]
  pub async fn set_motion_lines_uniform_spacetime(
    &self,
    count: usize,
    sampling_rate: f32,
    fps: f32,
  ) -> Result<(), JsValue> {
    self
      .set_motion_lines(MotionLineConfig {
        seed_selection:    SeedSelectionAlgorithm::UniformSpacetimeVertices {
          count,
          sampling_rate,
        },
        frames_per_second: fps,
      })
      .await
  }

  async fn set_motion_lines(&self, config: MotionLineConfig) -> Result<(), JsValue> {
    let Some(queue) = self.queue() else {
      return Err(JsValue::from_str("viewer runtime has stopped"));
    };
    let result = queue
      .configure_web_motion_lines(self.id, config)
      .await
      .map_err(|_| JsValue::from_str("motion-line configuration was cancelled"))?;
    result.map_err(|error| JsValue::from_str(&error.to_string()))
  }

  #[wasm_bindgen(js_name = clearMotionLines)]
  pub fn clear_motion_lines(&self) {
    if let Some(queue) = self.queue() {
      queue.clear_web_motion_lines(self.id);
    }
  }

  pub fn set_camera(
    &self,
    eye: js_sys::Array,
    target: js_sys::Array,
    up: js_sys::Array,
    fov: f32,
  ) -> Result<(), JsValue> {
    fn vec3(a: &js_sys::Array) -> Vec3 {
      Vec3::new(
        a.get(0).as_f64().unwrap_or(0.) as f32,
        a.get(1).as_f64().unwrap_or(0.) as f32,
        a.get(2).as_f64().unwrap_or(0.) as f32,
      )
    }
    if let Some(queue) = self.queue() {
      let camera = CameraConfig {
        eye: vec3(&eye),
        target: vec3(&target),
        up: vec3(&up),
        fov,
      };
      // A scripted path can submit this once per animation frame. Keep
      // only its newest pose until winit is ready to render, rather than
      // queueing a backlog that would make the animation lag behind.
      queue.set_web_camera(self.id, camera);
    }
    Ok(())
  }

  /// Sets a camera offset relative to the current animated mesh center. The
  /// renderer resolves the center after advancing the pose, so scripted
  /// camera paths follow root motion without requiring JavaScript to guess the
  /// mesh position or clipping range.
  #[wasm_bindgen(js_name = setCameraFollowMesh)]
  pub fn set_camera_follow_mesh(
    &self,
    offset: js_sys::Array,
    up: js_sys::Array,
    fov: f32,
  ) -> Result<(), JsValue> {
    fn vec3(a: &js_sys::Array) -> Vec3 {
      Vec3::new(
        a.get(0).as_f64().unwrap_or(0.) as f32,
        a.get(1).as_f64().unwrap_or(0.) as f32,
        a.get(2).as_f64().unwrap_or(0.) as f32,
      )
    }
    if let Some(queue) = self.queue() {
      queue.set_web_camera_follow_mesh(
        self.id,
        CameraFollowConfig {
          offset: vec3(&offset),
          up: vec3(&up),
          fov,
        },
      );
    }
    Ok(())
  }

  /// Resolves when all work queued before this call can execute on a viewer.
  #[wasm_bindgen(js_name = ready)]
  pub async fn ready(&self) -> Result<(), JsValue> {
    let Some(queue) = self.queue() else {
      return Err(JsValue::from_str("viewer runtime has stopped"));
    };
    queue
      .enqueue_web_with_result(self.id, |_| ())
      .await
      .map_err(|_| JsValue::from_str("viewer task was cancelled"))
  }

  pub fn dispose(&self) {
    RUNTIME.with(|runtime| {
      if let Some(runtime) = runtime.borrow().as_ref() {
        let _ = runtime.proxy.send_event(ApplicationEvent::Detach {
          id:         self.id,
          completion: None,
        });
        if self.independent {
          runtime.independent_ids.borrow_mut().remove(&self.id);
        }
      }
    });
  }

  /// Detaches this canvas and resolves only after the event loop has
  /// released the old wgpu surface and window.
  ///
  /// JavaScript must await this when switching slides. Merely sending a
  /// detach event and constructing another `ViewerHandle` immediately can
  /// overlap two WebGPU surface lifetimes and corrupt winit's web runner.
  #[wasm_bindgen(js_name = disposeAsync)]
  pub async fn dispose_async(&self) -> Result<(), JsValue> {
    let receiver = RUNTIME.with(|runtime| {
      let runtime = runtime.borrow();
      let runtime = runtime.as_ref()?;
      let (sender, receiver) = oneshot::channel();
      runtime
        .proxy
        .send_event(ApplicationEvent::Detach {
          id:         self.id,
          completion: Some(sender),
        })
        .ok()?;
      if self.independent {
        runtime.independent_ids.borrow_mut().remove(&self.id);
      }
      Some(receiver)
    });
    let Some(receiver) = receiver else {
      // The page-wide event loop may already be gone during page close.
      // In that case there is no remaining GPU resource to await.
      return Ok(());
    };
    receiver
      .await
      .map_err(|_| JsValue::from_str("viewer detach was cancelled"))
  }
}

#[wasm_bindgen(start)]
pub fn start() { console_error_panic_hook::set_once(); }

/// Converts Rust animation metadata into ordinary JavaScript objects without
/// requiring a serialization dependency in the shared renderer crate.
fn animation_infos_to_js(infos: &[AnimationInfo]) -> Result<JsValue, JsValue> {
  let animations = js_sys::Array::new();
  for info in infos {
    let animation = js_sys::Object::new();
    js_sys::Reflect::set(&animation, &"name".into(), &info.name.clone().into())?;
    js_sys::Reflect::set(&animation, &"duration".into(), &info.duration.into())?;
    let channels = js_sys::Array::new();
    for channel in &info.channels {
      let value = js_sys::Object::new();
      js_sys::Reflect::set(&value, &"node".into(), &channel.node.into())?;
      js_sys::Reflect::set(&value, &"property".into(), &channel.property.clone().into())?;
      js_sys::Reflect::set(
        &value,
        &"interpolation".into(),
        &channel.interpolation.clone().into(),
      )?;
      channels.push(&value);
    }
    js_sys::Reflect::set(&animation, &"channels".into(), &channels)?;
    animations.push(&animation);
  }
  Ok(animations.into())
}
