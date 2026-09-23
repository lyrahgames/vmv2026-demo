//! The readiness-aware task runtime for the viewer application.
//!
//! A script may run before winit has created a window and before WebGPU has
//! created a device. It therefore cannot safely receive `&mut Viewer`.
//! [`TaskQueue`] records work now and runs it only after the viewer is ready.

#[cfg(target_arch = "wasm32")]
use crate::camera::CameraFollowConfig;
use crate::{
  camera::CameraConfig, common::*, interaction::Interaction, mesh::Mesh,
  motion_lines::MotionLineConfig, scene::AnimatedScene,
  viewer::{MotionLineRenderStyle, Viewer},
};
use futures::{
  Future, FutureExt, executor::LocalPool, future::LocalBoxFuture, task::LocalSpawnExt,
};
use std::collections::{HashMap, VecDeque};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, closure::Closure};
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::{WindowAttributesExtWebSys, WindowExtWebSys};
use winit::{
  application::ApplicationHandler,
  event::WindowEvent,
  event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy},
  window::{Window, WindowAttributes, WindowId},
};

/// A mutation guaranteed to run with an initialized graphics context.
pub type ViewerAction = Box<dyn FnOnce(&mut Viewer) + 'static>;

#[cfg(target_arch = "wasm32")]
enum WebCameraPose {
  Fixed(CameraConfig),
  FollowMesh(CameraFollowConfig),
}

/// A deferred action or asynchronous routine. Async routines receive a client,
/// never a viewer reference, so they cannot keep GPU state borrowed over await.
enum ViewerTask {
  Action {
    id:     Option<u64>,
    action: ViewerAction,
  },
  Async {
    id:      Option<u64>,
    routine: Box<dyn FnOnce(ViewerTaskClient) -> LocalBoxFuture<'static, ()> + 'static>,
  },
}

struct TaskQueueState {
  tasks:             VecDeque<ViewerTask>,
  // Viewer construction needs one mesh. `set_mesh` records this seed as well
  // as its regular replacement action, leaving all other actions opaque.
  initial_meshes:    HashMap<Option<u64>, Mesh>,
  proxy:             Option<EventLoopProxy<ApplicationEvent>>,
  // Keep only the newest camera pose for the active web component. Camera
  // animation is a state stream, not a FIFO command stream: an intermediate
  // pose has no value after JavaScript has submitted a newer one.
  #[cfg(target_arch = "wasm32")]
  latest_web_camera: HashMap<u64, WebCameraPose>,
}

/// A clonable producer for viewer work.
///
/// It can be populated before `run_app`; [`App::bind_queue`] supplies the
/// event-loop wakeup later. This lets a Lua startup script configure the app
/// before the native event loop exists.
#[derive(Clone)]
pub struct TaskQueue {
  state: Rc<RefCell<TaskQueueState>>,
}

/// Capability given to async routines after the viewer is ready.
///
/// The client can queue a short follow-up graphics action after a future
/// resolves, but intentionally cannot access `Viewer` directly.
#[derive(Clone)]
pub struct ViewerTaskClient {
  queue: TaskQueue,
}

impl TaskQueue {
  /// Creates an empty queue that is safe to fill before application startup.
  pub fn new() -> Self {
    Self {
      state: Rc::new(RefCell::new(TaskQueueState {
        tasks: VecDeque::new(),
        initial_meshes: HashMap::new(),
        proxy: None,
        #[cfg(target_arch = "wasm32")]
        latest_web_camera: HashMap::new(),
      })),
    }
  }

  fn wake(&self) {
    if let Some(proxy) = self.state.borrow().proxy.as_ref() {
      // Failure means winit is shutting down, so the work can no longer
      // be executed and reporting an error to a teardown caller is noise.
      let _ = proxy.send_event(ApplicationEvent::RunTasks);
    }
  }

  fn push(&self, task: ViewerTask) {
    self.state.borrow_mut().tasks.push_back(task);
    self.wake();
  }

  fn bind(&self, proxy: EventLoopProxy<ApplicationEvent>) {
    self.state.borrow_mut().proxy = Some(proxy);
  }

  fn initial_mesh(&self, #[cfg(target_arch = "wasm32")] active_id: Option<u64>) -> Option<Mesh> {
    self
      .state
      .borrow()
      .initial_meshes
      .get(&{
        #[cfg(target_arch = "wasm32")]
        {
          active_id
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
          None
        }
      })
      .cloned()
  }

  #[cfg(not(target_arch = "wasm32"))]
  fn pop(&self) -> Option<ViewerTask> { self.state.borrow_mut().tasks.pop_front() }

  #[cfg(target_arch = "wasm32")]
  fn pop_for(&self, id: u64) -> Option<ViewerTask> {
    let mut state = self.state.borrow_mut();
    let index = state.tasks.iter().position(|task| match task {
      ViewerTask::Action { id: task_id, .. } | ViewerTask::Async { id: task_id, .. } => {
        *task_id == Some(id)
      }
    })?;
    state.tasks.remove(index)
  }

  /// Queues fire-and-forget graphics work.
  pub fn enqueue(&self, action: impl FnOnce(&mut Viewer) + 'static) {
    self.push(ViewerTask::Action {
      id:     None,
      action: Box::new(action),
    });
  }

  /// Queues a mesh replacement and makes it available for first construction.
  pub fn set_mesh(&self, mesh: Mesh) {
    self
      .state
      .borrow_mut()
      .initial_meshes
      .insert(None, mesh.clone());
    self.enqueue(move |viewer| viewer.mesh_replace(mesh));
  }

  /// Queues a complete glTF scene while exposing its first frame as the
  /// initialization mesh. Viewer construction can therefore begin before
  /// the queued scene replacement runs, without losing any commands.
  pub fn set_scene(&self, scene: AnimatedScene) {
    let initial_mesh = scene.initial_mesh();
    self
      .state
      .borrow_mut()
      .initial_meshes
      .insert(None, initial_mesh.clone());
    // The first frame is already needed to construct the viewer. Carry it
    // into the queued replacement as well, so loading an FBX does not
    // sample its complete surface twice before the first draw.
    self.enqueue(move |viewer| viewer.scene_replace_with_mesh(scene, initial_mesh));
  }

  pub fn reset_camera(&self) {
    self.enqueue(|viewer| viewer.reset_camera());
  }

  /// Frames the complete selected animation while keeping the camera fixed.
  pub fn frame_animation(&self) {
    self.enqueue(|viewer| viewer.frame_animation());
  }

  /// Selects a loaded glTF animation without starting playback.
  pub fn select_animation(&self, index: usize) {
    self.enqueue(move |viewer| viewer.select_animation(index));
  }

  /// Starts the selected glTF animation.
  pub fn play_animation(&self) {
    self.enqueue(|viewer| viewer.play_animation());
  }

  /// Pauses the selected glTF animation.
  pub fn pause_animation(&self) {
    self.enqueue(|viewer| viewer.pause_animation());
  }

  /// Sets the selected animation's current time in seconds.
  pub fn set_animation_time(&self, time: f32) {
    self.enqueue(move |viewer| viewer.set_animation_time(time));
  }

  /// Sets the playback multiplier for the selected animation.
  pub fn set_animation_speed(&self, speed: f32) {
    self.enqueue(move |viewer| viewer.set_animation_speed(speed));
  }

  /// Configures the seed-selection and uniform sampling stage on native.
  /// Browser callers use the result-bearing variant below so JavaScript can
  /// report invalid settings or GPU-size limits to the slide component.
  pub fn configure_motion_lines(&self, config: MotionLineConfig) {
    self.enqueue(move |viewer| {
      if let Err(error) = viewer.configure_motion_lines(config) {
        eprintln!("could not configure motion lines: {error:#}");
      }
    });
  }

  pub fn clear_motion_lines(&self) {
    self.enqueue(|viewer| viewer.clear_motion_lines());
  }

  /// Selects the line fragment style without retracing the animation.
  pub fn set_motion_line_style(&self, style: MotionLineRenderStyle) {
    self.enqueue(move |viewer| viewer.set_motion_line_style(style));
  }

  /// Queues a native diagnostic of CPU RSS and viewer-owned GPU allocations.
  /// The action is deferred because Lua scripts run before the wgpu viewer is
  /// initialized.
  #[cfg(not(target_arch = "wasm32"))]
  pub fn print_memory_usage(&self, label: String) {
    self.enqueue(move |viewer| viewer.print_memory_usage(&label));
  }

  pub fn set_camera(&self, camera: CameraConfig) {
    self.enqueue(move |viewer| viewer.set_camera(camera));
  }

  /// Queues a normalized linear-RGB background color for the native viewer.
  pub fn set_background_color(&self, color: [f32; 3]) {
    self.enqueue(move |viewer| viewer.set_background_color(color));
  }

  pub fn hide_scene(&self) {
    self.enqueue(|viewer| viewer.hide_scene());
  }

  pub fn show_scene(&self) {
    self.enqueue(|viewer| viewer.show_scene());
  }

  /// Queues a one-shot native screenshot of the next rendered frame.
  #[cfg(not(target_arch = "wasm32"))]
  pub fn request_screenshot(&self, path: String) {
    self.enqueue(move |viewer| viewer.request_screenshot(path));
  }

  pub fn resize(&self, size: winit::dpi::PhysicalSize<u32>) {
    self.enqueue(move |viewer| viewer.resize(size));
  }

  /// Queues graphics work and returns a future for its result.
  ///
  /// Do not await this from a Lua startup script: it would wait for the
  /// event loop that has not started yet. JS or Rust async callers can await
  /// it once the application is running.
  pub fn enqueue_with_result<T: 'static>(
    &self,
    action: impl FnOnce(&mut Viewer) -> T + 'static,
  ) -> oneshot::Receiver<T> {
    let (send, receive) = oneshot::channel();
    self.enqueue(move |viewer| {
      let _ = send.send(action(viewer));
    });
    receive
  }

  /// Starts a routine only after readiness and returns its result receiver.
  ///
  /// A routine receives [`ViewerTaskClient`] instead of `&mut Viewer`; after
  /// awaiting external work it can use that client to enqueue a continuation.
  pub fn enqueue_async<T: 'static, F, Fut>(&self, routine: F) -> oneshot::Receiver<T>
  where
    F: FnOnce(ViewerTaskClient) -> Fut + 'static,
    Fut: Future<Output = T> + 'static,
  {
    let (send, receive) = oneshot::channel();
    self.push(ViewerTask::Async {
      id:      None,
      routine: Box::new(move |client| {
        async move {
          let _ = send.send(routine(client).await);
        }
        .boxed_local()
      }),
    });
    receive
  }

  /// Tags browser work with its component ID, preventing late callbacks from
  /// an old Slidev component from changing the currently visible viewer.
  #[cfg(target_arch = "wasm32")]
  pub fn enqueue_web(&self, id: u64, action: impl FnOnce(&mut Viewer) + 'static) {
    self.push(ViewerTask::Action {
      id:     Some(id),
      action: Box::new(action),
    });
  }

  /// Replaces the pending web camera pose without waking the event loop.
  ///
  /// JavaScript may call this at 60 FPS. Rendering already schedules its next
  /// frame, so sending an EventLoopProxy user event for every camera update
  /// would only add latency and duplicate wakeups. The render callback takes
  /// this value immediately before encoding its GPU commands.
  #[cfg(target_arch = "wasm32")]
  pub fn set_web_camera(&self, id: u64, camera: CameraConfig) {
    self
      .state
      .borrow_mut()
      .latest_web_camera
      .insert(id, WebCameraPose::Fixed(camera));
  }

  /// Replaces the pending web camera offset with a pose that follows the
  /// currently animated mesh. The viewer resolves the target and clipping
  /// planes after the current animation pose has been uploaded.
  #[cfg(target_arch = "wasm32")]
  pub fn set_web_camera_follow_mesh(&self, id: u64, camera: CameraFollowConfig) {
    self
      .state
      .borrow_mut()
      .latest_web_camera
      .insert(id, WebCameraPose::FollowMesh(camera));
  }

  /// Browser-scoped equivalent of [`Self::set_background_color`].
  #[cfg(target_arch = "wasm32")]
  pub fn set_web_background_color(&self, id: u64, color: [f32; 3]) {
    self.enqueue_web(id, move |viewer| viewer.set_background_color(color));
  }

  /// Takes the most recent pose for the active component immediately before
  /// a frame is rendered. This keeps camera animation off the task queue.
  #[cfg(target_arch = "wasm32")]
  fn take_web_camera(&self, active_id: u64) -> Option<WebCameraPose> {
    self.state.borrow_mut().latest_web_camera.remove(&active_id)
  }

  /// Drops a pose that belongs to a detached canvas.
  #[cfg(target_arch = "wasm32")]
  fn clear_web_camera(&self, id: Option<u64>) {
    let mut state = self.state.borrow_mut();
    if let Some(id) = id {
      state.latest_web_camera.remove(&id);
    } else {
      state.latest_web_camera.clear();
    }
  }

  /// Drops browser work that belongs to the canvas being detached.
  ///
  /// The queue is shared by all slide components because winit owns one
  /// page-wide event loop. A slide transition must therefore remove its
  /// queued closures and initialization mesh immediately; waiting for a
  /// later redraw would retain large FBX/glTF allocations unnecessarily and
  /// could let an obsolete readiness sender survive into the next slide.
  #[cfg(target_arch = "wasm32")]
  fn clear_web_state(&self, id: Option<u64>) {
    let mut state = self.state.borrow_mut();
    match id {
      Some(id) => {
        state.tasks.retain(|task| match task {
          ViewerTask::Action { id: task_id, .. } | ViewerTask::Async { id: task_id, .. } => {
            *task_id != Some(id)
          }
        });
        state.initial_meshes.remove(&Some(id));
        state.latest_web_camera.remove(&id);
      }
      None => {
        state.tasks.retain(|task| match task {
          ViewerTask::Action { id: task_id, .. } | ViewerTask::Async { id: task_id, .. } => {
            task_id.is_none()
          }
        });
        state.initial_meshes.retain(|key, _| key.is_none());
        state.latest_web_camera.clear();
      }
    }
  }

  #[cfg(target_arch = "wasm32")]
  pub fn set_web_mesh(&self, id: u64, mesh: Mesh) {
    self
      .state
      .borrow_mut()
      .initial_meshes
      .insert(Some(id), mesh.clone());
    self.enqueue_web(id, move |viewer| viewer.mesh_replace(mesh));
  }

  /// Browser-scoped equivalent of [`Self::set_scene`].
  #[cfg(target_arch = "wasm32")]
  pub fn set_web_scene(&self, id: u64, scene: AnimatedScene) {
    let initial_mesh = scene.initial_mesh();
    self
      .state
      .borrow_mut()
      .initial_meshes
      .insert(Some(id), initial_mesh.clone());
    self.push(ViewerTask::Action {
      id:     Some(id),
      action: Box::new(move |viewer| viewer.scene_replace_with_mesh(scene, initial_mesh)),
    });
    self.wake();
  }

  /// Web equivalent of [`Self::enqueue_with_result`]. If its slide is no
  /// longer active, the sender is dropped and JavaScript receives a rejected
  /// promise rather than accidentally observing the next slide's viewer.
  #[cfg(target_arch = "wasm32")]
  pub fn enqueue_web_with_result<T: 'static>(
    &self,
    id: u64,
    action: impl FnOnce(&mut Viewer) -> T + 'static,
  ) -> oneshot::Receiver<T> {
    let (send, receive) = oneshot::channel();
    self.push(ViewerTask::Action {
      id:     Some(id),
      action: Box::new(move |viewer| {
        let _ = send.send(action(viewer));
      }),
    });
    receive
  }

  /// Browser-scoped animation controls discard commands from stale slides.
  #[cfg(target_arch = "wasm32")]
  pub fn select_web_animation(&self, id: u64, index: usize) {
    self.enqueue_web(id, move |viewer| viewer.select_animation(index));
  }

  #[cfg(target_arch = "wasm32")]
  pub fn play_web_animation(&self, id: u64) {
    self.enqueue_web(id, |viewer| viewer.play_animation());
  }

  #[cfg(target_arch = "wasm32")]
  pub fn pause_web_animation(&self, id: u64) {
    self.enqueue_web(id, |viewer| viewer.pause_animation());
  }

  #[cfg(target_arch = "wasm32")]
  pub fn set_web_animation_time(&self, id: u64, time: f32) {
    self.enqueue_web(id, move |viewer| viewer.set_animation_time(time));
  }

  #[cfg(target_arch = "wasm32")]
  pub fn set_web_animation_speed(&self, id: u64, speed: f32) {
    self.enqueue_web(id, move |viewer| viewer.set_animation_speed(speed));
  }

  /// Browser-scoped motion-line configuration with an acknowledgement for
  /// JavaScript callers. The acknowledgement means the CPU preparation and
  /// compute dispatch have been queued on the active viewer.
  #[cfg(target_arch = "wasm32")]
  pub fn configure_web_motion_lines(
    &self,
    id: u64,
    config: MotionLineConfig,
  ) -> oneshot::Receiver<Result<()>> {
    self.enqueue_web_with_result(id, move |viewer| viewer.configure_motion_lines(config))
  }

  #[cfg(target_arch = "wasm32")]
  pub fn clear_web_motion_lines(&self, id: u64) {
    self.enqueue_web(id, |viewer| viewer.clear_motion_lines());
  }

  #[cfg(target_arch = "wasm32")]
  pub fn set_web_motion_line_style(&self, id: u64, style: MotionLineRenderStyle) {
    self.enqueue_web(id, move |viewer| viewer.set_motion_line_style(style));
  }
}

impl Default for TaskQueue {
  fn default() -> Self {
    Self::new()
  }
}

/// Executes a native Lua preview without creating a window or a wgpu surface.
///
/// Preview scripts are intentionally synchronous: they load the scene and
/// queue all render configuration before the one offscreen frame is rendered.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_headless(
  queue: TaskQueue,
  size: winit::dpi::PhysicalSize<u32>,
) -> Result<()> {
  let mesh = queue
    .initial_mesh()
    .ok_or_else(|| anyhow!("preview script did not load a mesh or scene"))?;
  let mut viewer = pollster::block_on(Viewer::new_headless(size, mesh))?;
  while let Some(task) = queue.pop() {
    match task {
      ViewerTask::Action { action, .. } => action(&mut viewer),
      ViewerTask::Async { .. } => {
        bail!("asynchronous tasks are not supported by headless preview scripts")
      }
    }
  }
  viewer.render();
  if !viewer.screenshot_complete() {
    bail!("headless preview script did not request a screenshot")
  }
  Ok(())
}

impl ViewerTaskClient {
  /// Schedules a graphics-side continuation of an async routine.
  pub fn enqueue(&self, action: impl FnOnce(&mut Viewer) + 'static) {
    self.queue.enqueue(action);
  }

  /// Uses a mesh obtained asynchronously as the next viewer mesh.
  pub fn set_mesh(&self, mesh: Mesh) {
    self.queue.set_mesh(mesh);
  }
}

/// Winit user events contain no closures. Closures stay in [`TaskQueue`],
/// while web lifecycle events retain their Slidev component identity.
pub enum ApplicationEvent {
  RunTasks,
  Dispose,
  #[cfg(target_arch = "wasm32")]
  RedrawAll,
  #[cfg(target_arch = "wasm32")]
  Attach {
    id:          u64,
    canvas:      HtmlCanvasElement,
    // Ordinary slides reuse one viewer.  A component sets this only when
    // it explicitly asks for an additional, simultaneous viewer.
    independent: bool,
  },
  #[cfg(target_arch = "wasm32")]
  Detach {
    id:         u64,
    // Browser disposal is asynchronous from JavaScript's perspective.
    // The acknowledgement is sent only after winit has dropped the
    // surface and window, so the next canvas cannot race that teardown.
    completion: Option<oneshot::Sender<()>>,
  },
}

// Winit installs one no-argument `requestAnimationFrame` closure for every
// canvas-backed window. That is convenient for a normal desktop window, but
// it is fragile when Slidev removes and recreates several canvases: a callback
// that was already queued in the browser can outlive the winit closure that
// owned it and fail as an "indirect call to null". The viewer does not need
// one RAF callback per surface. A single page-level callback sends a harmless
// user event and the application renders every live viewer.
//
// The callback is intentionally retained for the lifetime of the WASM module.
// It therefore cannot be dropped while the browser still has a frame callback
// queued. Individual viewers are still detached and release their WebGPU
// surfaces normally; only the tiny scheduler remains alive.
#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_FRAME_CALLBACK: RefCell<Option<Closure<dyn FnMut(f64)>>> =
        const { RefCell::new(None) };
}

#[cfg(target_arch = "wasm32")]
/// Starts the browser's one stable frame pump.
pub(crate) fn start_web_frame_loop(proxy: EventLoopProxy<ApplicationEvent>) {
  WEB_FRAME_CALLBACK.with(|callback_slot| {
    // ViewerHandle::new may be called by multiple slide components.  Only
    // the first call owns the global callback; later calls simply reuse it.
    if callback_slot.borrow().is_some() {
      return;
    }

    let callback = Closure::wrap(Box::new(move |_timestamp: f64| {
      let _ = proxy.send_event(ApplicationEvent::RedrawAll);

      // Keep requesting frames through the same retained closure.  The
      // immutable borrow ends before the browser invokes this callback
      // again on a later animation frame.
      WEB_FRAME_CALLBACK.with(|callback_slot| {
        if let Some(callback) = callback_slot.borrow().as_ref() {
          if let Some(window) = web_sys::window() {
            let _ = window.request_animation_frame(callback.as_ref().unchecked_ref());
          }
        }
      });
    }) as Box<dyn FnMut(f64)>);

    *callback_slot.borrow_mut() = Some(callback);
    if let Some(callback) = callback_slot.borrow().as_ref() {
      if let Some(window) = web_sys::window() {
        let _ = window.request_animation_frame(callback.as_ref().unchecked_ref());
      }
    }
  });
}

#[cfg(target_arch = "wasm32")]
struct WebViewerState {
  window:         Option<Arc<Window>>,
  window_id:      Option<WindowId>,
  viewer:         Option<Viewer>,
  interaction:    Interaction,
  initializing:   bool,
  generation:     u64,
  last_frame:     Option<f64>,
  pending_canvas: Option<HtmlCanvasElement>,
  independent:    bool,
}

struct AppState {
  queue:        TaskQueue,
  window:       Option<Arc<Window>>,
  attributes:   WindowAttributes,
  viewer:       Option<Viewer>,
  #[cfg(not(target_arch = "wasm32"))]
  interaction:  Interaction,
  initializing: bool,
  window_id:    Option<WindowId>,
  #[cfg(not(target_arch = "wasm32"))]
  generation:   u64,
  // Futures can finish synchronously while they are being polled. Keeping
  // the executor behind its own cell lets such a future call back into App
  // without holding the larger AppState borrow.
  async_pool:   Rc<RefCell<LocalPool>>,
  active_async: Rc<Cell<usize>>,
  // Animation time is advanced from actual redraw callbacks. This keeps
  // playback tied to rendered frames instead of task-queue traffic.
  #[cfg(not(target_arch = "wasm32"))]
  last_frame:   Option<Instant>,
  #[cfg(target_arch = "wasm32")]
  last_frame:   Option<f64>,
  #[cfg(target_arch = "wasm32")]
  active_id:    Option<u64>,
  #[cfg(target_arch = "wasm32")]
  web_viewers:  HashMap<u64, WebViewerState>,
}

/// The winit application that owns the one concrete viewer.
pub struct App {
  state: Rc<RefCell<AppState>>,
}

impl App {
  pub fn new(queue: TaskQueue) -> Self {
    Self {
      state: Rc::new(RefCell::new(AppState {
        queue,
        window: None,
        attributes: Window::default_attributes(),
        viewer: None,
        #[cfg(not(target_arch = "wasm32"))]
        interaction: Interaction::default(),
        initializing: false,
        window_id: None,
        #[cfg(not(target_arch = "wasm32"))]
        generation: 0,
        async_pool: Rc::new(RefCell::new(LocalPool::new())),
        active_async: Rc::new(Cell::new(0)),
        last_frame: None,
        #[cfg(target_arch = "wasm32")]
        active_id: None,
        #[cfg(target_arch = "wasm32")]
        web_viewers: HashMap::new(),
      })),
    }
  }

  pub fn set_window_attributes(&mut self, attributes: WindowAttributes) {
    self.state.borrow_mut().attributes = attributes;
  }

  /// Registers the first browser canvas before `spawn_app` starts winit.
  ///
  /// The first web viewer cannot rely on a user event sent immediately after
  /// `spawn_app`: winit may enter its initial `resumed` callback before that
  /// event is dispatched.  Keeping the pending canvas in AppState from the
  /// beginning makes the initial window creation deterministic.  Later
  /// canvases are still installed through `ApplicationEvent::Attach`.
  #[cfg(target_arch = "wasm32")]
  pub fn set_initial_web_canvas(&mut self, id: u64, canvas: HtmlCanvasElement, independent: bool) {
    self.state.borrow_mut().web_viewers.insert(
      id,
      WebViewerState {
        window: None,
        window_id: None,
        viewer: None,
        interaction: Interaction::default(),
        initializing: false,
        generation: 0,
        last_frame: None,
        pending_canvas: Some(canvas),
        independent,
      },
    );
    if !independent {
      self.state.borrow_mut().active_id = Some(id);
    }
  }

  /// Connects existing task producers after the winit event loop is built.
  pub fn bind_queue(&self, event_loop: &winit::event_loop::EventLoop<ApplicationEvent>) {
    self.state.borrow().queue.bind(event_loop.create_proxy());
  }

  fn request_redraw(&self) {
    #[cfg(target_arch = "wasm32")]
    {
      // Web rendering is driven by the retained page-level RAF callback
      // above. Calling Window::request_redraw here would re-enable
      // winit's per-canvas RAF closures and reintroduce the callback
      // lifetime race that makes the first independent viewer vanish.
      return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
      let window = self.state.borrow().window.clone();
      if let Some(window) = window {
        window.request_redraw();
      }
    }
  }

  /// Tears down the complete native application.
  ///
  /// `Viewer` owns a wgpu `Surface` which refers to its winit window.  An
  /// explicit order is required on native close: relying on `AppState` field
  /// order can destroy the window first and make backend shutdown crash.
  fn shutdown(&self) {
    let mut state = self.state.borrow_mut();
    drop(state.viewer.take());
    drop(state.window.take());
    state.window_id = None;
    state.initializing = false;
    state.last_frame = None;
  }

  /// Runs all actions now eligible for execution and launches eligible futures.
  #[cfg(not(target_arch = "wasm32"))]
  fn drain_tasks(&self) {
    // A user event can arrive immediately after Lua queues a
    // mesh, but before `Viewer::new` has completed.  Leave every task in
    // FIFO order in that case; consuming it here was what caused the web
    // promise to be cancelled and left the canvas empty.
    if self.state.borrow().viewer.is_none() {
      self.begin_initialization();
      return;
    }
    // Keep the queue borrow independent from AppState. The `while let`
    // form using `self.state.borrow()` directly keeps a Ref alive while
    // each task is executed and can conflict with the mutable viewer
    // borrow below.
    let queue = self.state.borrow().queue.clone();
    loop {
      let Some(task) = queue.pop() else {
        break;
      };
      match task {
        ViewerTask::Action {
          id: task_id,
          action,
        } => {
          #[cfg(not(target_arch = "wasm32"))]
          let _ = task_id;
          #[cfg(target_arch = "wasm32")]
          if task_id.is_some() && task_id != self.state.borrow().active_id {
            continue;
          }
          let mut viewer = {
            let mut state = self.state.borrow_mut();
            state.viewer.take()
          };
          if let Some(mut viewer_value) = viewer.take() {
            action(&mut viewer_value);
            self.state.borrow_mut().viewer = Some(viewer_value);
          }
        }
        ViewerTask::Async {
          id: task_id,
          routine,
        } => {
          #[cfg(not(target_arch = "wasm32"))]
          let _ = task_id;
          #[cfg(target_arch = "wasm32")]
          if task_id.is_some() && task_id != self.state.borrow().active_id {
            continue;
          }
          let (client, active, spawner) = {
            let state = self.state.borrow();
            let client = ViewerTaskClient {
              queue: state.queue.clone(),
            };
            let active = state.active_async.clone();
            // End the LocalPool borrow before the surrounding
            // AppState borrow is dropped at the block boundary.
            let spawner = {
              let pool = state.async_pool.borrow();
              pool.spawner()
            };
            (client, active, spawner)
          };
          active.set(active.get() + 1);
          spawner
            .spawn_local(async move {
              routine(client).await;
              active.set(active.get().saturating_sub(1));
            })
            .expect("local task executor unexpectedly stopped");
        }
      }
    }
    self.request_redraw();
  }

  #[cfg(target_arch = "wasm32")]
  fn drain_tasks(&self) {
    self.drain_web_tasks();
  }

  #[cfg(target_arch = "wasm32")]
  fn drain_web_tasks(&self) {
    // Each viewer consumes only its own tagged actions.  This prevents a
    // slow second viewer from blocking the first viewer's ready promise.
    let ids: Vec<u64> = self
      .state
      .borrow()
      .web_viewers
      .iter()
      .filter_map(|(id, slot)| slot.viewer.as_ref().map(|_| *id))
      .collect();

    // Clone the queue before entering the loop. Borrowing it through
    // `self.state.borrow()` in the `while let` condition keeps the
    // AppState Ref alive for the whole loop body; the first action then
    // panics when it tries to take the viewer mutably.
    let queue = self.state.borrow().queue.clone();
    for id in ids {
      while let Some(task) = queue.pop_for(id) {
        match task {
          ViewerTask::Action { action, .. } => {
            // Temporarily remove the viewer from AppState before
            // invoking user-facing completion machinery.  Sending
            // an oneshot result can synchronously wake JavaScript,
            // which may call back into this event loop.  Keeping a
            // RefMut alive across `action` would panic on that
            // perfectly valid re-entry.
            let mut viewer = {
              let mut state = self.state.borrow_mut();
              state
                .web_viewers
                .get_mut(&id)
                .and_then(|slot| slot.viewer.take())
            };
            if let Some(mut viewer_value) = viewer.take() {
              action(&mut viewer_value);
              let mut state = self.state.borrow_mut();
              if let Some(slot) = state.web_viewers.get_mut(&id) {
                slot.viewer = Some(viewer_value);
              }
            }
          }
          ViewerTask::Async { routine, .. } => {
            let (client, active, spawner) = {
              let state = self.state.borrow();
              let client = ViewerTaskClient {
                queue: state.queue.clone(),
              };
              let active = state.active_async.clone();
              let spawner = state.async_pool.borrow().spawner();
              (client, active, spawner)
            };
            active.set(active.get() + 1);
            spawner
              .spawn_local(async move {
                routine(client).await;
                active.set(active.get().saturating_sub(1));
              })
              .expect("local task executor unexpectedly stopped");
          }
        }
      }
    }
    self.request_redraw();
  }

  fn poll_async_tasks(&self, event_loop: &ActiveEventLoop) {
    // Release the AppState borrow before polling. A future such as browser
    // viewer initialization can complete during this call and immediately
    // borrow AppState again from its completion callback.
    let async_pool = self.state.borrow().async_pool.clone();
    async_pool.borrow_mut().run_until_stalled();
    // Outstanding local futures need polling. Otherwise the next redraw
    // is explicitly requested by the previous redraw callback, allowing
    // the event loop to sleep between frames instead of spinning.
    let has_active_async = self.state.borrow().active_async.get() != 0;
    event_loop.set_control_flow(if has_active_async {
      ControlFlow::Poll
    } else {
      ControlFlow::Wait
    });
  }

  #[cfg(not(target_arch = "wasm32"))]
  fn begin_initialization(&self) {
    let (window, mesh, generation) = {
      let mut state = self.state.borrow_mut();
      if state.viewer.is_some() || state.initializing {
        return;
      }
      let Some(window) = state.window.clone() else {
        return;
      };
      let Some(mesh) = state.queue.initial_mesh() else {
        return;
      };
      state.initializing = true;
      (window, mesh, state.generation)
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
      let viewer =
        pollster::block_on(Viewer::new(window, mesh)).expect("GPU initialization failed");
      let mut state = self.state.borrow_mut();
      if state.generation == generation {
        state.viewer = Some(viewer);
      }
      state.initializing = false;
      drop(state);
      self.drain_tasks();
    }
  }

  #[cfg(target_arch = "wasm32")]
  fn begin_initialization(&self) {
    let ids: Vec<u64> = self.state.borrow().web_viewers.keys().copied().collect();
    for id in ids {
      self.begin_web_initialization(id);
    }
  }

  #[cfg(target_arch = "wasm32")]
  fn begin_web_initialization(&self, id: u64) {
    // Query the queue outside the AppState borrow. The queue has its own
    // RefCell, and keeping the surrounding state borrow alive across this
    // initialization boundary made browser callbacks re-entering App
    // panic with "RefCell already borrowed".
    let queue = self.state.borrow().queue.clone();
    let Some(mesh) = queue.initial_mesh(Some(id)) else {
      return;
    };
    let (window, generation) = {
      let mut state = self.state.borrow_mut();
      let Some(slot) = state.web_viewers.get_mut(&id) else {
        return;
      };
      if slot.viewer.is_some() || slot.initializing || slot.window.is_none() {
        return;
      }
      slot.initializing = true;
      (slot.window.clone().unwrap(), slot.generation)
    };

    let state_ref = self.state.clone();
    wasm_bindgen_futures::spawn_local(async move {
      match Viewer::new(window, mesh).await {
        Ok(viewer) => {
          let app = App {
            state: state_ref.clone(),
          };
          let accepted = {
            let mut state = state_ref.borrow_mut();
            if let Some(slot) = state.web_viewers.get_mut(&id) {
              if slot.generation == generation {
                slot.viewer = Some(viewer);
                slot.initializing = false;
                true
              } else {
                false
              }
            } else {
              false
            }
          };
          if accepted {
            app.drain_web_tasks();
          }
        }
        Err(error) => {
          web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(&error.to_string()));
          if let Some(slot) = state_ref.borrow_mut().web_viewers.get_mut(&id) {
            if slot.generation == generation {
              slot.initializing = false;
            }
          }
        }
      }
    });
  }

  #[cfg(target_arch = "wasm32")]
  fn apply_web_event(&self, event: ApplicationEvent) {
    match event {
      ApplicationEvent::Attach {
        id,
        canvas,
        independent,
      } => {
        let mut reusable_viewer = None;
        let queue = self.state.borrow().queue.clone();
        let old_ids: Vec<u64> = {
          let state = self.state.borrow();
          if independent {
            Vec::new()
          } else {
            state
              .web_viewers
              .iter()
              .filter_map(|(old_id, slot)| (!slot.independent && *old_id != id).then_some(*old_id))
              .collect()
          }
        };

        // Ordinary slides are still a singleton: remove only the old
        // ordinary slot, but move its Viewer/device into the new slot
        // so slide navigation does not allocate another GPU device.
        for old_id in old_ids {
          queue.clear_web_state(Some(old_id));
          let mut old = self.state.borrow_mut().web_viewers.remove(&old_id).unwrap();
          if let Some(viewer) = old.viewer.as_mut() {
            viewer.hide_scene();
            viewer.detach_surface();
          }
          drop(old.window.take());
          reusable_viewer = old.viewer.take();
        }

        let mut state = self.state.borrow_mut();
        let generation = state
          .web_viewers
          .get(&id)
          .map(|slot| slot.generation.wrapping_add(1))
          .unwrap_or(1);
        state.web_viewers.insert(
          id,
          WebViewerState {
            window: None,
            window_id: None,
            viewer: reusable_viewer,
            interaction: Interaction::default(),
            initializing: false,
            generation,
            last_frame: None,
            pending_canvas: Some(canvas),
            independent,
          },
        );
        if !independent {
          state.active_id = Some(id);
        }
        drop(state);
        queue.clear_web_camera(Some(id));
      }
      ApplicationEvent::Detach { id, completion } => {
        let queue = self.state.borrow().queue.clone();
        queue.clear_web_state(Some(id));
        let removed = self.state.borrow_mut().web_viewers.remove(&id);
        if let Some(mut slot) = removed {
          if let Some(viewer) = slot.viewer.as_mut() {
            viewer.hide_scene();
            viewer.detach_surface();
          }
          drop(slot.window.take());
        }
        if self.state.borrow().active_id == Some(id) {
          self.state.borrow_mut().active_id = None;
        }
        if let Some(completion) = completion {
          let _ = completion.send(());
        }
      }
      _ => {}
    }
  }

  #[cfg(target_arch = "wasm32")]
  fn process_pending_canvas(&self, event_loop: &ActiveEventLoop) {
    let ids: Vec<u64> = self
      .state
      .borrow()
      .web_viewers
      .iter()
      .filter_map(|(id, slot)| slot.pending_canvas.is_some().then_some(*id))
      .collect();
    for id in ids {
      let Some(canvas) = self
        .state
        .borrow_mut()
        .web_viewers
        .get_mut(&id)
        .and_then(|slot| slot.pending_canvas.take())
      else {
        continue;
      };
      let scale = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0);
      let client_width = canvas.client_width();
      let client_height = canvas.client_height();
      let has_seeded_backing_size = canvas.width() > 1
                && canvas.height() > 1
                // A newly created HTML canvas has this browser default.  Any
                // other value was written by ObjViewer after layout settled.
                && (canvas.width() != 300 || canvas.height() != 150);
      let size = if has_seeded_backing_size {
        // Preserve the exact physical size measured by JavaScript.
        // Recomputing it from CSS pixels here can round a high-DPI
        // grid cell differently from the ResizeObserver measurement.
        winit::dpi::PhysicalSize::new(canvas.width(), canvas.height())
      } else if client_width > 0 && client_height > 0 {
        // Prefer the live CSS layout size.  ObjViewer seeds the
        // backing store before this callback, but CSS remains the
        // source of truth when a slide is resized between frames.
        winit::dpi::PhysicalSize::new(
          ((client_width as f64) * scale).round() as u32,
          ((client_height as f64) * scale).round() as u32,
        )
      } else if canvas.width() > 1 && canvas.height() > 1 {
        // If winit is called before the browser exposes client
        // dimensions, retain the physical size seeded by JavaScript
        // instead of falling back to a destructive 1x1 surface.
        winit::dpi::PhysicalSize::new(canvas.width(), canvas.height())
      } else {
        // A hidden/zero-layout canvas is not ready for a wgpu
        // surface.  Leave it pending; the JavaScript layout gate
        // normally prevents this path, while this guard protects
        // against a late Slidev transition.
        let mut state = self.state.borrow_mut();
        if let Some(slot) = state.web_viewers.get_mut(&id) {
          slot.pending_canvas = Some(canvas);
        }
        continue;
      };
      canvas.set_width(size.width);
      canvas.set_height(size.height);
      let window = event_loop
        .create_window(
          Window::default_attributes()
            // Slidev owns focus while it changes slides.  Letting
            // winit focus an embedded canvas during construction
            // synchronously dispatches blur/focus events into the
            // same web event-loop callback that is constructing
            // the window.  With two independent canvases that
            // re-entry can call a wasm closure after winit has
            // replaced it, producing "indirect call to null".
            // A viewer does not need initial focus; a pointer
            // event can still be delivered to the canvas later.
            .with_active(false)
            // Do not make winit call `canvas.focus()` from its
            // pointer handlers.  The presentation controls focus
            // and the viewer only needs pointer coordinates.  It
            // also prevents a browser focus event from re-entering
            // winit while a second independent window is being
            // registered.
            .with_prevent_default(false)
            // Do not pass `with_inner_size` for an existing DOM
            // canvas. Winit translates that attribute into fixed
            // CSS pixel width/height declarations, which fights
            // Slidev's responsive grid and leaves the canvas
            // visually stretched after a slide transition. The
            // physical `canvas.width/height` values above are the
            // actual WebGPU backing size; the DOM keeps its own
            // responsive CSS layout.
            .with_canvas(Some(canvas)),
        )
        .expect("window creation failed");
      let window = Arc::new(window);
      // A surface attachment can cross into browser/wgpu code. Take the
      // viewer out before doing that so no RefMut<AppState> survives the
      // callback boundary.
      let mut viewer = {
        let mut state = self.state.borrow_mut();
        state
          .web_viewers
          .get_mut(&id)
          .and_then(|slot| slot.viewer.take())
      };
      if let Some(viewer_value) = viewer.as_mut() {
        viewer_value
          .attach_window(window.clone())
          .expect("failed to attach viewer to browser canvas");
      }
      let mut state = self.state.borrow_mut();
      if let Some(slot) = state.web_viewers.get_mut(&id) {
        slot.window_id = Some(window.id());
        slot.window = Some(window);
        slot.viewer = viewer;
      }
    }
  }

  #[cfg(target_arch = "wasm32")]
  fn web_window_event(&self, window_id: WindowId, event: WindowEvent) {
    let id = self
      .state
      .borrow()
      .web_viewers
      .iter()
      .find_map(|(id, slot)| (slot.window_id == Some(window_id)).then_some(*id));
    let Some(id) = id else {
      return;
    };

    match event {
      WindowEvent::CloseRequested => {
        self.apply_web_event(ApplicationEvent::Detach {
          id,
          completion: None,
        });
      }
      WindowEvent::Resized(size) => {
        // The browser-side resize observer sizes the HTML canvas in
        // physical pixels (`client_size * devicePixelRatio`). The
        // web implementation of winit also emits `Resized`, but its
        // value can describe the logical CSS size on some browsers.
        // Using that value directly would silently downgrade the
        // WebGPU surface after the canvas was already configured for
        // high-DPI rendering. Read the actual canvas backing store
        // instead; it is the source of truth for the surface.
        let surface_size = {
          let state = self.state.borrow();
          state
            .web_viewers
            .get(&id)
            .and_then(|slot| slot.window.as_ref())
            .and_then(|window| window.canvas())
            .map(|canvas| winit::dpi::PhysicalSize::new(canvas.width(), canvas.height()))
            .filter(|canvas_size| canvas_size.width > 0 && canvas_size.height > 0)
            .unwrap_or(size)
        };

        let mut viewer = {
          let mut state = self.state.borrow_mut();
          state
            .web_viewers
            .get_mut(&id)
            .and_then(|slot| slot.viewer.take())
        };
        if let Some(mut viewer_value) = viewer.take() {
          viewer_value.resize(surface_size);
          if let Some(slot) = self.state.borrow_mut().web_viewers.get_mut(&id) {
            slot.viewer = Some(viewer_value);
          }
        }
      }
      WindowEvent::RedrawRequested => {
        // Keep this branch for an already queued winit callback, but
        // use the same borrow-safe renderer as the stable page-level
        // RAF path.
        self.render_web_viewer(id);
      }
      event => {
        let mut viewer = {
          let mut state = self.state.borrow_mut();
          state
            .web_viewers
            .get_mut(&id)
            .and_then(|slot| slot.viewer.take())
        };
        if let Some(mut viewer_value) = viewer.take() {
          if let Some(slot) = self.state.borrow_mut().web_viewers.get_mut(&id) {
            slot.interaction.event(&event, &mut viewer_value.camera);
            slot.viewer = Some(viewer_value);
          }
        }
      }
    }
  }

  /// Renders one browser viewer without holding an AppState borrow across
  /// camera updates, GPU submission, or JavaScript-visible completion.
  #[cfg(target_arch = "wasm32")]
  fn render_web_viewer(&self, id: u64) {
    let now = web_sys::window()
      .and_then(|window| window.performance())
      .map(|performance| performance.now() / 1000.0)
      .unwrap_or_else(|| js_sys::Date::now() / 1000.0);
    // Clone the queue before taking the AppState borrow. The queue has its
    // own RefCell, and keeping an AppState borrow alive while taking the
    // latest camera pose was the source of the earlier re-entry panic.
    let queue = self.state.borrow().queue.clone();
    let camera = queue.take_web_camera(id);
    let (mut viewer, delta) = {
      let mut state = self.state.borrow_mut();
      let Some(slot) = state.web_viewers.get_mut(&id) else {
        return;
      };
      let delta = slot
        .last_frame
        .replace(now)
        .map(|previous| (now - previous).max(0.0).min(0.1) as f32)
        .unwrap_or(0.0);
      (slot.viewer.take(), delta)
    };
    if let Some(mut viewer_value) = viewer.take() {
      if let Some(camera) = camera {
        match camera {
          WebCameraPose::Fixed(camera) => viewer_value.set_camera(camera),
          WebCameraPose::FollowMesh(camera) => viewer_value.set_camera_follow_mesh(camera),
        }
      }
      viewer_value.update_animation(delta);
      viewer_value.render();
      let mut state = self.state.borrow_mut();
      if let Some(slot) = state.web_viewers.get_mut(&id) {
        slot.viewer = Some(viewer_value);
      }
    }
  }

  /// Renders all attached browser canvases in one event-loop turn. This is
  /// what makes the explicit dual-view example genuinely simultaneous: the
  /// first viewer is no longer dependent on whichever canvas happened to
  /// request the next winit redraw.
  #[cfg(target_arch = "wasm32")]
  fn render_web_viewers(&self) {
    let ids: Vec<u64> = self.state.borrow().web_viewers.keys().copied().collect();
    for id in ids {
      self.render_web_viewer(id);
    }
  }
}

impl ApplicationHandler<ApplicationEvent> for App {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    #[cfg(target_arch = "wasm32")]
    {
      self.process_pending_canvas(event_loop);
      self.begin_initialization();
      return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    let mut state = self.state.borrow_mut();
    #[cfg(not(target_arch = "wasm32"))]
    if state.window.is_none() {
      let window = event_loop
        .create_window(state.attributes.clone())
        .expect("window creation failed");
      state.window_id = Some(window.id());
      state.window = Some(Arc::new(window));
    }
    #[cfg(not(target_arch = "wasm32"))]
    drop(state);
    #[cfg(not(target_arch = "wasm32"))]
    self.begin_initialization();
  }

  fn user_event(&mut self, event_loop: &ActiveEventLoop, event: ApplicationEvent) {
    #[cfg(target_arch = "wasm32")]
    if matches!(
      &event,
      ApplicationEvent::Attach { .. } | ApplicationEvent::Detach { .. }
    ) {
      self.apply_web_event(event);
      return;
    }
    #[cfg(target_arch = "wasm32")]
    if matches!(&event, ApplicationEvent::RedrawAll) {
      self.render_web_viewers();
      return;
    }
    match event {
      ApplicationEvent::RunTasks => self.drain_tasks(),
      ApplicationEvent::Dispose => {
        self.shutdown();
        event_loop.exit();
      }
      #[cfg(target_arch = "wasm32")]
      ApplicationEvent::Attach { .. }
      | ApplicationEvent::Detach { .. }
      | ApplicationEvent::RedrawAll => unreachable!(),
    }
  }

  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    window_id: WindowId,
    event: WindowEvent,
  ) {
    #[cfg(target_arch = "wasm32")]
    let _ = event_loop;
    #[cfg(target_arch = "wasm32")]
    {
      self.web_window_event(window_id, event);
      return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
      let mut state = self.state.borrow_mut();
      if state.window_id != Some(window_id) {
        return;
      }
      match event {
        WindowEvent::CloseRequested => {
          // Release the wgpu surface while its native window still
          // exists, then let winit complete the close operation.
          drop(state);
          self.shutdown();
          event_loop.exit();
        }
        WindowEvent::Resized(size) => {
          if let Some(viewer) = state.viewer.as_mut() {
            viewer.resize(size)
          }
        }
        WindowEvent::RedrawRequested => {
          // `Instant::now()` is not implemented by the wasm32-unknown-
          // unknown standard library. Use the browser performance clock
          // there, while retaining the monotonic native implementation.
          #[cfg(not(target_arch = "wasm32"))]
          let delta = {
            let now = Instant::now();
            state
              .last_frame
              .replace(now)
              .map(|previous| now.duration_since(previous).as_secs_f32().min(0.1))
              .unwrap_or(0.0)
          };
          #[cfg(target_arch = "wasm32")]
          let delta = {
            let now = web_sys::window()
              .and_then(|window| window.performance())
              .map(|performance| performance.now() / 1000.0)
              .unwrap_or_else(|| js_sys::Date::now() / 1000.0);
            state
              .last_frame
              .replace(now)
              .map(|previous| (now - previous).max(0.0).min(0.1) as f32)
              .unwrap_or(0.0)
          };
          #[cfg(target_arch = "wasm32")]
          let camera = state.queue.take_web_camera(state.active_id);
          if let Some(viewer) = state.viewer.as_mut() {
            // Apply only the newest JavaScript pose immediately before
            // rendering. This avoids an event-loop wakeup and a task
            // allocation for every animation frame.
            #[cfg(target_arch = "wasm32")]
            if let Some(camera) = camera {
              match camera {
                WebCameraPose::Fixed(camera) => viewer.set_camera(camera),
                WebCameraPose::FollowMesh(camera) => viewer.set_camera_follow_mesh(camera),
              }
            }
            // The animation update happens immediately before the
            // draw, so a selected glTF clip remains independent from
            // Lua/JavaScript command timing.
            viewer.update_animation(delta);
            viewer.render();
            #[cfg(not(target_arch = "wasm32"))]
            if viewer.screenshot_complete() {
              drop(state);
              event_loop.exit();
              return;
            }
            // A viewer is an animation surface, not a one-shot paint
            // surface. Request the next frame from the frame callback
            // itself; this remains reliable when no task is queued.
            state
              .window
              .as_ref()
              .expect("window disappeared during redraw")
              .request_redraw();
          }
        }
        event => {
          let AppState {
            interaction,
            viewer,
            ..
          } = &mut *state;
          if let Some(viewer) = viewer.as_mut() {
            interaction.event(&event, &mut viewer.camera)
          }
        }
      }
    }
  }

  fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
    #[cfg(target_arch = "wasm32")]
    self.process_pending_canvas(event_loop);
    self.begin_initialization();
    #[cfg(not(target_arch = "wasm32"))]
    let has_native_viewer = self.state.borrow().viewer.is_some();
    #[cfg(not(target_arch = "wasm32"))]
    if has_native_viewer {
      self.drain_tasks();
    }
    #[cfg(target_arch = "wasm32")]
    let has_web_viewer = {
      let state = self.state.borrow();
      state.web_viewers.values().any(|slot| slot.viewer.is_some())
    };
    #[cfg(target_arch = "wasm32")]
    if has_web_viewer {
      self.drain_tasks();
    }
    self.poll_async_tasks(event_loop);
  }
}
