//! GPU resource ownership and the actual mesh render pass.
//!
//! `Viewer` intentionally owns the wgpu surface, device, queue, pipeline, and
//! buffers together.  The event-loop layer only changes camera/mesh state and
//! calls [`Viewer::render`] when winit asks for a frame.

use crate::{
  camera::{Camera, CameraConfig, CameraFollowConfig},
  common::*,
  mesh::{Mesh, SkinTransform, SkinnedVertex, Vertex},
  motion_lines::{
    MotionLineConfig, PoseSamples, SpacetimeSelectionOptions, prepare_pose_samples, prepare_samples,
  },
  scene::{AnimatedGpuMesh, AnimatedScene, AnimationInfo},
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;
use winit::window::Window;

/// Maximum number of Catmull–Rom subdivisions available for one uniform
/// segment. The post-process pass begins with up to eight steps and doubles
/// that count when its flatness test calls for more samples.
const MAX_MOTION_LINE_SUBDIVISIONS: u32 = 16;
/// Hard cap for the persistent adaptively sampled output. The cap is applied
/// before allocation by reducing the largest per-segment subdivision level.
const MAX_MOTION_LINE_POSTPROCESS_BYTES: u64 = 64 * 1024 * 1024;
/// Temporary trace inputs (uniform trajectories plus sampled pose streams)
/// coexist with the output while extraction runs. They are destroyed after
/// submission, but this peak cap prevents a large animation from creating a
/// transient allocation spike on a browser adapter.
const MAX_MOTION_LINE_PEAK_BYTES: u64 = 128 * 1024 * 1024;
/// Maximum temporary position-stream allocation for spacetime seed selection.
/// Unlike the final trajectory bundle this stream contains only one vec4 per
/// vertex/time pair and is destroyed immediately after the seed indices have
/// been generated.
const MAX_SPACETIME_SEED_POSITION_BYTES: u64 = 64 * 1024 * 1024;
/// The greedy spacetime selector needs one dependent GPU submission per
/// promoted seed. This guard prevents an accidental request for tens of
/// thousands of seeds from flooding the queue with dependent submissions.
const MAX_SPACETIME_SELECTION_PASSES: u64 = 4096;
/// Repeated three-point stencils suppress pose-scale jitter in the average
/// camera target while preserving the broad root motion of the animation.
const CAMERA_TARGET_STENCIL_PASSES: usize = 12;
/// Four samples per pixel is the portable WebGPU/native MSAA level. The
/// renderer checks every format used by the scene and OIT passes before
/// enabling it, falling back to one sample only on adapters that cannot
/// resolve all of those formats.
const PREFERRED_MSAA_SAMPLE_COUNT: u32 = 4;

/// Selects the fragment shader used for an already extracted motion-line
/// bundle. Switching styles deliberately does not retrace trajectories.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MotionLineRenderStyle {
  /// The original presentation stroke. This remains the default.
  #[default]
  Teaser,
  /// The arc-length dashed stroke adapted from `paper-compasso`.
  Dashed,
  /// A continuous opaque dark gray stroke.
  FullTrajectory,
  /// The same ribbon clipped to the current time window.
  WindowedFullTrajectory,
  /// The teaser colormap and time window on an opaque stroke.
  UnweightedTeaser,
}

/// GPU buffers for an animated scene.  Geometry and indices are immutable;
/// only the compact transform palette is uploaded as the pose changes.
struct AnimatedGpuState {
  vertex:            wgpu::Buffer,
  index:             wgpu::Buffer,
  palette:           wgpu::Buffer,
  // Kept alive because the bind group references this immutable storage
  // buffer; the render pass reaches it indirectly through `bind`.
  _morph_positions:  wgpu::Buffer,
  morph_weights:     wgpu::Buffer,
  bind:              wgpu::BindGroup,
  transforms:        Vec<SkinTransform>,
  morph_weights_cpu: Vec<f32>,
  vertex_count:      usize,
  count:             u32,
}

/// A translucent copy of the animated surface sampled at one fixed time.
struct PhantomPose {
  _palette:       wgpu::Buffer,
  _morph_weights: wgpu::Buffer,
  bind:           wgpu::BindGroup,
  _opacity:       wgpu::Buffer,
  opacity_bind:   wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MotionLineParams {
  seed_count:          u32,
  sample_count:        u32,
  output_stride:       u32,
  vertex_word_stride:  u32,
  palette_stride:      u32,
  morph_weight_stride: u32,
  max_subdivisions:    u32,
  samples_per_second:  f32,
  duration:            f32,
  tolerance:           f32,
  reserved:            [f32; 3],
  // WGSL uniform structures are rounded up to a 16-byte size. Keep the
  // explicit tail padding in the Rust representation so the upload has the
  // same 64-byte footprint on every backend.
  _padding:            [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpacetimeSeedTraceParams {
  vertex_count:        u32,
  sample_count:        u32,
  vertex_word_stride:  u32,
  palette_stride:      u32,
  morph_weight_stride: u32,
  samples_per_second:  f32,
  duration:            f32,
  _padding:            [u32; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpacetimeSeedSelectionParams {
  vertex_count:   u32,
  sample_count:   u32,
  seed_count:     u32,
  stage:          u32,
  selected_count: u32,
  importance: u32,
  stochastic: u32,
  extended: u32,
}

/// Visual parameters for the screen-space speedline bundle.
///
/// The first vector follows the old teaser's temporal weighting, the second
/// controls the view-aligned strip and fragment depth halo, and the last
/// vector is the physical canvas size used for pixel-constant widths. Keeping
/// these values in one small uniform makes the effect work identically for
/// native windows and the slide deck's WebGPU canvas.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MotionLineStyle {
  timing:   [f32; 4], // now, visible tail duration, characteristic length, reserved
  widths:   [f32; 4], // strip width px, max depth halo, Compasso-style flag, reserved
  viewport: [f32; 4], // physical width px, physical height px
}

/// One sample in either the uniformly traced or adaptively resampled bundle.
/// The fields mirror the WGSL storage layout and are kept as four-component
/// values so the GPU can address every field with the same 16-byte stride.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TrajectorySample {
  position: [f32; 4],
  velocity: [f32; 4],
  normal:   [f32; 4],
  metadata: [f32; 4], // x = time, y = arc length
}

/// GPU-resident result of tracing followed by adaptive post-processing.
///
/// `trajectories` is laid out seed-major with a fixed per-seed output stride;
/// `counts` records how much of each strip the adaptive pass actually wrote.
/// The uniform trace and sampled pose streams are transient compute inputs,
/// so only this final bundle remains resident for rendering.
struct MotionLineGpuState {
  // The selected vertex IDs also drive the live seed-point render pass.
  trajectories:  wgpu::Buffer,
  counts:        wgpu::Buffer,
  params:        wgpu::Buffer,
  style:         wgpu::Buffer,
  seeds:         wgpu::Buffer,
  line_bind:     wgpu::BindGroup,
  seed_bind:     wgpu::BindGroup,
  seed_count:    u32,
  output_stride: u32,
}

impl MotionLineGpuState {
  fn destroy(self) {
    self.trajectories.destroy();
    self.counts.destroy();
    self.params.destroy();
    self.style.destroy();
    self.seeds.destroy();
  }

  fn gpu_buffer_bytes(&self) -> u64 {
    self.trajectories.size()
      + self.counts.size()
      + self.params.size()
      + self.style.size()
      + self.seeds.size()
  }
}

/// Render targets for weighted-blended order-independent transparency.
///
/// Motion lines are first accumulated into a premultiplied-color/weight
/// buffer and a multiplicative revealage buffer. A later fullscreen pass
/// resolves those targets over the already-rendered mesh. The targets are
/// recreated with the surface because their dimensions are presentation-size
/// dependent.
struct MotionLineOitTargets {
  // Single-sampled textures are the inputs to the fullscreen composite pass.
  accumulation:           wgpu::Texture,
  accumulation_view:      wgpu::TextureView,
  revealage:              wgpu::Texture,
  revealage_view:         wgpu::TextureView,
  // The line pass writes these multisampled attachments and resolves into the
  // textures above. They remain optional for the safe one-sample fallback.
  accumulation_msaa:      Option<wgpu::Texture>,
  accumulation_msaa_view: Option<wgpu::TextureView>,
  revealage_msaa:         Option<wgpu::Texture>,
  revealage_msaa_view:    Option<wgpu::TextureView>,
  sample_count:           u32,
  width:                  u32,
  height:                 u32,
}

impl MotionLineOitTargets {
  fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, sample_count: u32) -> Self {
    let size = wgpu::Extent3d {
      width:                 config.width.max(1),
      height:                config.height.max(1),
      depth_or_array_layers: 1,
    };
    let accumulation = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("motion-line OIT accumulation"),
      size,
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Rgba16Float,
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
      view_formats: &[],
    });
    let accumulation_view = accumulation.create_view(&Default::default());
    let (accumulation_msaa, accumulation_msaa_view) = if sample_count > 1 {
      let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("motion-line OIT accumulation MSAA"),
        size,
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
      });
      let view = texture.create_view(&Default::default());
      (Some(texture), Some(view))
    } else {
      (None, None)
    };
    let revealage = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("motion-line OIT revealage"),
      size,
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::R8Unorm,
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
      view_formats: &[],
    });
    let revealage_view = revealage.create_view(&Default::default());
    let (revealage_msaa, revealage_msaa_view) = if sample_count > 1 {
      let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("motion-line OIT revealage MSAA"),
        size,
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
      });
      let view = texture.create_view(&Default::default());
      (Some(texture), Some(view))
    } else {
      (None, None)
    };
    Self {
      accumulation,
      accumulation_view,
      revealage,
      revealage_view,
      accumulation_msaa,
      accumulation_msaa_view,
      revealage_msaa,
      revealage_msaa_view,
      sample_count,
      width: config.width.max(1),
      height: config.height.max(1),
    }
  }

  fn destroy(&self) {
    self.accumulation.destroy();
    self.revealage.destroy();
    if let Some(texture) = &self.accumulation_msaa {
      texture.destroy();
    }
    if let Some(texture) = &self.revealage_msaa {
      texture.destroy();
    }
  }

  fn gpu_bytes(&self) -> u64 {
    // Rgba16Float uses eight bytes per pixel; R8Unorm uses one. The resolved
    // textures are always present; MSAA attachments multiply the extra copy
    // by their actual sample count.
    let pixels = self.width as u64 * self.height as u64;
    pixels * (8 + 1) * (1 + u64::from(self.sample_count > 1) * self.sample_count as u64)
  }
}

impl AnimatedGpuState {
  /// Explicitly releases the large scene buffers before a replacement scene
  /// is allocated. WebGPU may otherwise keep the old allocation alive until
  /// a later browser GC/device tick, causing a transient memory spike.
  fn destroy(self) {
    self.vertex.destroy();
    self.index.destroy();
    self.palette.destroy();
    self._morph_positions.destroy();
    self.morph_weights.destroy();
  }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
  // The vertex shader multiplies each world-space position by this matrix.
  view_proj:   [[f32; 4]; 4],
  // The fragment shader uses the camera position for view-dependent toon and
  // silhouette shading.
  camera:      [f32; 4],
  // Non-sRGB browser surface formats display linear shader values too dark.
  // A value of one enables shader-side sRGB encoding for that fallback.
  encode_srgb: [f32; 4],
}

/// A pending camera orbit whose look-at target is supplied by the current
/// animated mesh pose rather than by the scripting layer.
#[derive(Clone, Copy)]
struct ActiveCameraFollow {
  offset: Vec3,
  up:     Vec3,
  fov:    f32,
}

/// Smooth camera target motion extracted from the average position of all
/// surface vertices over one animation. The path is intentionally separate
/// from the instantaneous pose bounds: an elbow or leg changing the AABB must
/// not make the camera target jump.
struct CameraTargetPath {
  animation: Option<usize>,
  duration:  f32,
  targets:   Vec<Vec3>,
  radius:    f32,
}

impl CameraTargetPath {
  fn sample(&self, time: f32) -> Vec3 {
    if self.targets.len() <= 1 || self.duration <= 0.0 {
      return self.targets.first().copied().unwrap_or(Vec3::ZERO);
    }
    let wrapped = time.rem_euclid(self.duration);
    let sample_time = if time > 0.0 && wrapped == 0.0 {
      self.duration
    } else {
      wrapped
    };
    let normalized = (sample_time / self.duration).clamp(0.0, 1.0);
    let position = normalized * (self.targets.len() - 1) as f32;
    let segment = position.floor() as usize;
    let factor = position - segment as f32;
    let p0 = self.targets[segment.saturating_sub(1)];
    let p1 = self.targets[segment];
    let p2 = self.targets[(segment + 1).min(self.targets.len() - 1)];
    let p3 = self.targets[(segment + 2).min(self.targets.len() - 1)];
    // Uniform Catmull–Rom interpolation smooths the average trajectory while
    // avoiding the frame-to-frame target jumps caused by pose bounds.
    0.5
      * ((2.0 * p1)
        + (-p0 + p2) * factor
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * factor * factor
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * factor * factor * factor)
  }
}

/// All GPU state needed to render one mesh into one window/canvas.
pub struct Viewer {
  /// Public because input handling and scripted paths both update it.
  pub camera: Camera,
  // CPU copy retained for bounds-based camera reset and replacement.
  mesh: Mesh,
  // A scene is present only for glTF assets. OBJ replacement clears it so
  // stale animation state can never affect a later static mesh.
  scene: Option<AnimatedScene>,
  animation_index: Option<usize>,
  animation_time: f32,
  animation_speed: f32,
  animation_playing: bool,
  // The shared viewer survives slide transitions, but its old scene must not
  // remain visible while the next slide is loading.
  scene_visible: bool,
  camera_follow: Option<ActiveCameraFollow>,
  camera_target_path: Option<CameraTargetPath>,
  // The instance is retained on wasm so the existing device can create a
  // surface for each newly visible Slidev canvas. Native applications never
  // reattach a surface after startup.
  #[cfg(target_arch = "wasm32")]
  instance: wgpu::Instance,
  // `'static` is valid because the Arc<Window> passed to create_surface is
  // retained by AppState for at least as long as this surface.
  surface: Option<wgpu::Surface<'static>>,
  // Native preview rendering uses an offscreen texture instead of a surface.
  #[cfg(not(target_arch = "wasm32"))]
  headless_target: Option<wgpu::Texture>,
  device: wgpu::Device,
  queue: wgpu::Queue,
  config: wgpu::SurfaceConfiguration,
  pipeline: wgpu::RenderPipeline,
  animated_pipeline: wgpu::RenderPipeline,
  phantom_pipeline: wgpu::RenderPipeline,
  phantom_opacity_layout: wgpu::BindGroupLayout,
  motion_line_pipeline: wgpu::RenderPipeline,
  motion_line_full_trajectory_pipeline: wgpu::RenderPipeline,
  motion_line_windowed_full_trajectory_pipeline: wgpu::RenderPipeline,
  motion_line_unweighted_pipeline: wgpu::RenderPipeline,
  motion_line_dashed_pipeline: wgpu::RenderPipeline,
  seed_point_pipeline: wgpu::RenderPipeline,
  motion_composite_pipeline: wgpu::RenderPipeline,
  motion_trace_pipeline: wgpu::ComputePipeline,
  motion_post_pipeline: wgpu::ComputePipeline,
  motion_seed_trace_pipeline: wgpu::ComputePipeline,
  motion_seed_selection_pipeline: wgpu::ComputePipeline,
  vertex: wgpu::Buffer,
  index: wgpu::Buffer,
  count: u32,
  uniform: wgpu::Buffer,
  bind: wgpu::BindGroup,
  skin_layout: wgpu::BindGroupLayout,
  motion_trace_layout: wgpu::BindGroupLayout,
  motion_post_layout: wgpu::BindGroupLayout,
  motion_seed_trace_layout: wgpu::BindGroupLayout,
  motion_seed_selection_layout: wgpu::BindGroupLayout,
  motion_line_layout: wgpu::BindGroupLayout,
  seed_point_layout: wgpu::BindGroupLayout,
  motion_composite_layout: wgpu::BindGroupLayout,
  motion_composite_bind: wgpu::BindGroup,
  animated: Option<AnimatedGpuState>,
  phantoms: Vec<PhantomPose>,
  motion_line_config: Option<MotionLineConfig>,
  motion_line_style: MotionLineRenderStyle,
  motion_line_opacity: f32,
  motion_lines_visible: bool,
  seed_points_visible: bool,
  motion_lines: Option<MotionLineGpuState>,
  last_motion_line_peak_bytes: u64,
  // MSAA color is resolved into the acquired surface texture. The depth
  // attachment and every geometry pipeline use the same sample count.
  sample_count: u32,
  msaa_color: Option<wgpu::TextureView>,
  depth: wgpu::TextureView,
  motion_oit: MotionLineOitTargets,
  // This remains constant for a surface configuration, but is written with
  // every frame beside the camera data for a simple, portable uniform ABI.
  encode_srgb: f32,
  // Linear RGB clear color selected by the native Lua or browser script.
  // Keeping this in the viewer rather than the surface configuration makes
  // the same script API work on native sRGB and browser UNORM surfaces.
  background_color: [f32; 3],
  #[cfg(not(target_arch = "wasm32"))]
  // Native preview scripts may request one readback from the next rendered
  // offscreen frame. This stays out of the browser build, where the fallback
  // assets are deliberately not generated at runtime.
  screenshot_path: Option<std::path::PathBuf>,
  #[cfg(not(target_arch = "wasm32"))]
  screenshot_complete: bool,
}

impl Drop for Viewer {
  fn drop(&mut self) {
    // WebGPU resources are released asynchronously by the browser. A
    // normal Rust drop only removes wgpu's handles; it does not force the
    // browser-side GPUDevice to release all buffers before the next slide
    // creates another device. Explicit destruction is essential for the
    // slide deck, which replaces animated models repeatedly.
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(target) = self.headless_target.take() {
      target.destroy();
    }
    self.device.destroy();
  }
}

impl Viewer {
  /// Returns the actual physical backing-store size of a web canvas.
  ///
  /// winit's web `Window::inner_size()` is populated by its asynchronous
  /// `ResizeObserver`.  During the same callback in which a canvas window is
  /// created it can therefore still report `0x0`, even though the DOM
  /// canvas already has the correct physical width and height.  Starting a
  /// surface with that transient value is the reason a canvas can render at
  /// a visibly poor resolution until a later resize event arrives.
  #[cfg(target_arch = "wasm32")]
  fn initial_surface_size(window: &Window) -> winit::dpi::PhysicalSize<u32> {
    window
      .canvas()
      .map(|canvas| winit::dpi::PhysicalSize::new(canvas.width(), canvas.height()))
      .filter(|size| size.width != 0 && size.height != 0)
      .unwrap_or_else(|| window.inner_size())
  }

  /// Creates a surface-backed viewer for the interactive native and web applications.
  pub async fn new(window: Arc<Window>, mesh: Mesh) -> Result<Self> {
    #[cfg(target_arch = "wasm32")]
    let size = Self::initial_surface_size(&window);
    #[cfg(not(target_arch = "wasm32"))]
    let size = window.inner_size();

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window)?;
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference:       wgpu::PowerPreference::HighPerformance,
        compatible_surface:     Some(&surface),
        force_fallback_adapter: false,
      })
      .await?;
    let (device, queue) = adapter
      .request_device(&wgpu::DeviceDescriptor {
        label:             Some("device"),
        required_features: wgpu::Features::empty(),
        required_limits:   wgpu::Limits::default(),
        memory_hints:      wgpu::MemoryHints::Performance,
        trace:             wgpu::Trace::Off,
      })
      .await?;
    let caps = surface.get_capabilities(&adapter);
    let format = caps
      .formats
      .iter()
      .copied()
      .find(|f| f.is_srgb())
      .or_else(|| caps.formats.first().copied())
      .ok_or_else(|| anyhow!("WebGPU surface reported no supported formats"))?;
    let encode_srgb = (!format.is_srgb()) as u8 as f32;
    let sample_count = choose_msaa_sample_count(&adapter, format);
    let alpha_mode = caps
      .alpha_modes
      .first()
      .copied()
      .ok_or_else(|| anyhow!("WebGPU surface reported no alpha modes"))?;
    let config = wgpu::SurfaceConfiguration {
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
      format,
      width: size.width.max(1),
      height: size.height.max(1),
      present_mode: wgpu::PresentMode::Fifo,
      alpha_mode,
      view_formats: vec![],
      desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    Self::new_with_graphics(
      mesh,
      Some(surface),
      device,
      queue,
      config,
      format,
      encode_srgb,
      sample_count,
      Some(instance),
      None,
    )
  }

  /// Creates a viewer whose render target is an offscreen texture.
  ///
  /// This path deliberately never creates a winit window or a surface. It is
  /// used by the manual preview command and therefore works on CI runners
  /// without Wayland, X11, or a display server.
  #[cfg(not(target_arch = "wasm32"))]
  pub async fn new_headless(
    size: winit::dpi::PhysicalSize<u32>,
    mesh: Mesh,
  ) -> Result<Self> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter_options = |force_fallback_adapter| wgpu::RequestAdapterOptions {
      power_preference:       wgpu::PowerPreference::LowPower,
      compatible_surface:     None,
      force_fallback_adapter,
    };
    let adapter = match instance.request_adapter(&adapter_options(false)).await {
      Ok(adapter) => adapter,
      Err(_) => instance.request_adapter(&adapter_options(true)).await?,
    };
    let (device, queue) = adapter
      .request_device(&wgpu::DeviceDescriptor {
        label:             Some("headless preview device"),
        required_features: wgpu::Features::empty(),
        required_limits:   wgpu::Limits::default(),
        memory_hints:      wgpu::MemoryHints::Performance,
        trace:             wgpu::Trace::Off,
      })
      .await?;

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let config = wgpu::SurfaceConfiguration {
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
      format,
      width: size.width.max(1),
      height: size.height.max(1),
      present_mode: wgpu::PresentMode::Fifo,
      alpha_mode: wgpu::CompositeAlphaMode::Opaque,
      view_formats: vec![],
      desired_maximum_frame_latency: 2,
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("headless preview target"),
      size: wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count: 1,
      dimension: wgpu::TextureDimension::D2,
      format,
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
      view_formats: &[],
    });
    let sample_count = choose_msaa_sample_count(&adapter, format);

    Self::new_with_graphics(
      mesh,
      None,
      device,
      queue,
      config,
      format,
      0.0,
      sample_count,
      None,
      Some(target),
    )
  }

  fn new_with_graphics(
    mesh: Mesh,
    surface: Option<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    format: wgpu::TextureFormat,
    encode_srgb: f32,
    sample_count: u32,
    _instance: Option<wgpu::Instance>,
    _headless_target: Option<wgpu::Texture>,
  ) -> Result<Self> {
    // The shader is embedded at compile time, keeping the WASM deployment
    // self-contained instead of requiring a second shader fetch.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  Some("mesh shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh.wgsl").into()),
    });
    let phantom_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("translucent phantom mesh shader"),
      source: wgpu::ShaderSource::Wgsl(concat!(
        include_str!("../shaders/mesh.wgsl"),
        "\n",
        include_str!("../shaders/mesh_phantom.wgsl"),
      ).into()),
    });
    let motion_trace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  Some("motion-line trace shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motion_trace.wgsl").into()),
    });
    let motion_seed_trace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  Some("motion-line spacetime seed trace shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motion_seed_trace.wgsl").into()),
    });
    let motion_seed_selection_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  Some("motion-line spacetime seed selection shader"),
      source: wgpu::ShaderSource::Wgsl(
        include_str!("../shaders/motion_seed_selection.wgsl").into(),
      ),
    });
    let motion_post_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  Some("motion-line post-process shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motion_postprocess.wgsl").into()),
    });
    let motion_line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  Some("motion-line render shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/motion_lines.wgsl").into()),
    });
    let seed_point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("seed-point ring shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/seed_points.wgsl").into()),
    });
    // The dashed fragment is kept in its own WGSL file, but shares the
    // geometry, OIT declarations, and helpers embedded above.
    let motion_line_dashed_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("motion-line dashed render shader"),
      source: wgpu::ShaderSource::Wgsl(concat!(
        include_str!("../shaders/motion_lines.wgsl"),
        "\n",
        include_str!("../shaders/motion_lines_dashed.wgsl"),
      ).into()),
    });
    let camera = Camera::new(config.width as f32 / config.height as f32);
    // Uniforms contain one 4x4 matrix and two vec4 values (96 bytes).
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("uniform"),
      size:               96,
      usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label:   Some("uniform layout"),
      entries: &[wgpu::BindGroupLayoutEntry {
        binding:    0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty:         wgpu::BindingType::Buffer {
          ty:                 wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size:   None,
        },
        count:      None,
      }],
    });
    // The bind group connects the uniform buffer to both shader stages.
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label:   Some("uniform bind"),
      layout:  &layout,
      entries: &[wgpu::BindGroupEntry {
        binding:  0,
        resource: uniform.as_entire_binding(),
      }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label:                Some("pipeline layout"),
      bind_group_layouts:   &[&layout],
      push_constant_ranges: &[],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label:         Some("mesh pipeline"),
      layout:        Some(&pipeline_layout),
      vertex:        wgpu::VertexState {
        module:              &shader,
        entry_point:         Some("vs_main"),
        buffers:             &[Vertex::layout()],
        compilation_options: Default::default(),
      },
      fragment:      Some(wgpu::FragmentState {
        module:              &shader,
        entry_point:         Some("fs_main"),
        targets:             &[Some(wgpu::ColorTargetState {
          format,
          blend: Some(wgpu::BlendState::REPLACE),
          write_mask: wgpu::ColorWrites::ALL,
        })],
        compilation_options: Default::default(),
      }),
      primitive:     wgpu::PrimitiveState::default(),
      depth_stencil: Some(wgpu::DepthStencilState {
        // Depth testing is required for the back faces of a 3D mesh to
        // be hidden behind front faces.
        format:              wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: true,
        depth_compare:       wgpu::CompareFunction::Less,
        stencil:             Default::default(),
        bias:                Default::default(),
      }),
      multisample:   multisample_state(sample_count),
      multiview:     None,
      cache:         None,
    });
    // The animated pipeline shares the camera uniform and fragment stage
    // with the static pipeline, but its vertex stage reads skin matrices
    // from a read-only storage buffer.
    let skin_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label:   Some("skin palette layout"),
      entries: &[
        wgpu::BindGroupLayoutEntry {
          binding:    0,
          visibility: wgpu::ShaderStages::VERTEX,
          ty:         wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size:   None,
          },
          count:      None,
        },
        wgpu::BindGroupLayoutEntry {
          binding:    1,
          visibility: wgpu::ShaderStages::VERTEX,
          ty:         wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size:   None,
          },
          count:      None,
        },
        wgpu::BindGroupLayoutEntry {
          binding:    2,
          visibility: wgpu::ShaderStages::VERTEX,
          ty:         wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size:   None,
          },
          count:      None,
        },
      ],
    });
    let animated_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label:                Some("animated pipeline layout"),
      bind_group_layouts:   &[&layout, &skin_layout],
      push_constant_ranges: &[],
    });
    let animated_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label:         Some("animated mesh pipeline"),
      layout:        Some(&animated_pipeline_layout),
      vertex:        wgpu::VertexState {
        module:              &shader,
        entry_point:         Some("vs_skinned"),
        buffers:             &[SkinnedVertex::layout()],
        compilation_options: Default::default(),
      },
      fragment:      Some(wgpu::FragmentState {
        module:              &shader,
        entry_point:         Some("fs_main"),
        targets:             &[Some(wgpu::ColorTargetState {
          format,
          blend: Some(wgpu::BlendState::REPLACE),
          write_mask: wgpu::ColorWrites::ALL,
        })],
        compilation_options: Default::default(),
      }),
      primitive:     wgpu::PrimitiveState::default(),
      depth_stencil: Some(wgpu::DepthStencilState {
        format:              wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: true,
        depth_compare:       wgpu::CompareFunction::Less,
        stencil:             Default::default(),
        bias:                Default::default(),
      }),
      multisample:   multisample_state(sample_count),
      multiview:     None,
      cache:         None,
    });
    let phantom_opacity_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("phantom opacity layout"),
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      }],
    });
    let phantom_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("translucent phantom pipeline layout"),
      bind_group_layouts: &[&layout, &skin_layout, &phantom_opacity_layout],
      push_constant_ranges: &[],
    });
    let phantom_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("translucent phantom mesh pipeline"),
      layout: Some(&phantom_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &phantom_shader,
        entry_point: Some("vs_skinned"),
        buffers: &[SkinnedVertex::layout()],
        compilation_options: Default::default(),
      },
      fragment: Some(wgpu::FragmentState {
        module: &phantom_shader,
        entry_point: Some("fs_phantom"),
        targets: &[Some(wgpu::ColorTargetState {
          format,
          blend: Some(wgpu::BlendState::ALPHA_BLENDING),
          write_mask: wgpu::ColorWrites::ALL,
        })],
        compilation_options: Default::default(),
      }),
      primitive: wgpu::PrimitiveState::default(),
      depth_stencil: Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::LessEqual,
        stencil: Default::default(),
        bias: Default::default(),
      }),
      multisample: multisample_state(sample_count),
      multiview: None,
      cache: None,
    });
    // Motion-line extraction has its own shader module and group-zero layout.
    // Keeping it separate from line rendering makes the required bind-group
    // index explicit to wgpu and avoids an unused group gap at dispatch.
    let motion_trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label:   Some("motion-line trace layout"),
      entries: &[
        storage_layout_entry(0, wgpu::ShaderStages::COMPUTE, true),
        storage_layout_entry(1, wgpu::ShaderStages::COMPUTE, true),
        storage_layout_entry(2, wgpu::ShaderStages::COMPUTE, true),
        storage_layout_entry(3, wgpu::ShaderStages::COMPUTE, true),
        storage_layout_entry(4, wgpu::ShaderStages::COMPUTE, true),
        storage_layout_entry(5, wgpu::ShaderStages::COMPUTE, false),
        uniform_layout_entry(6, wgpu::ShaderStages::COMPUTE),
      ],
    });
    let motion_seed_trace_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label:   Some("motion-line spacetime seed trace layout"),
        entries: &[
          storage_layout_entry(0, wgpu::ShaderStages::COMPUTE, true),
          storage_layout_entry(1, wgpu::ShaderStages::COMPUTE, true),
          storage_layout_entry(2, wgpu::ShaderStages::COMPUTE, true),
          storage_layout_entry(3, wgpu::ShaderStages::COMPUTE, true),
          storage_layout_entry(4, wgpu::ShaderStages::COMPUTE, false),
          uniform_layout_entry(5, wgpu::ShaderStages::COMPUTE),
        ],
      });
    let motion_seed_selection_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label:   Some("motion-line spacetime seed selection layout"),
        entries: &[
          storage_layout_entry(0, wgpu::ShaderStages::COMPUTE, true),
          storage_layout_entry(1, wgpu::ShaderStages::COMPUTE, false),
          storage_layout_entry(2, wgpu::ShaderStages::COMPUTE, false),
          storage_layout_entry(3, wgpu::ShaderStages::COMPUTE, false),
          storage_layout_entry(4, wgpu::ShaderStages::COMPUTE, false),
          uniform_layout_entry(5, wgpu::ShaderStages::COMPUTE),
        ],
      });
    let motion_post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label:   Some("motion-line post-process layout"),
      entries: &[
        storage_layout_entry(0, wgpu::ShaderStages::COMPUTE, true),
        storage_layout_entry(1, wgpu::ShaderStages::COMPUTE, false),
        storage_layout_entry(2, wgpu::ShaderStages::COMPUTE, false),
        uniform_layout_entry(3, wgpu::ShaderStages::COMPUTE),
      ],
    });
    let motion_line_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label:   Some("motion-line render layout"),
      entries: &[
        storage_layout_entry(0, wgpu::ShaderStages::VERTEX, true),
        uniform_layout_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
        storage_layout_entry(2, wgpu::ShaderStages::VERTEX, true),
        uniform_layout_entry(3, wgpu::ShaderStages::VERTEX_FRAGMENT),
      ],
    });
    let motion_composite_layout =
      device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label:   Some("motion-line OIT composite layout"),
        entries: &[
          texture_layout_entry(0, wgpu::ShaderStages::FRAGMENT),
          texture_layout_entry(1, wgpu::ShaderStages::FRAGMENT),
        ],
      });
    let motion_trace_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("motion-line trace pipeline layout"),
        bind_group_layouts:   &[&motion_trace_layout],
        push_constant_ranges: &[],
      });
    let motion_trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label:               Some("motion-line trace pipeline"),
      layout:              Some(&motion_trace_pipeline_layout),
      module:              &motion_trace_shader,
      entry_point:         Some("trace"),
      compilation_options: Default::default(),
      cache:               None,
    });
    let motion_seed_trace_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("motion-line spacetime seed trace pipeline layout"),
        bind_group_layouts:   &[&motion_seed_trace_layout],
        push_constant_ranges: &[],
      });
    let motion_seed_trace_pipeline =
      device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:               Some("motion-line spacetime seed trace pipeline"),
        layout:              Some(&motion_seed_trace_pipeline_layout),
        module:              &motion_seed_trace_shader,
        entry_point:         Some("trace"),
        compilation_options: Default::default(),
        cache:               None,
      });
    let motion_seed_selection_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("motion-line spacetime seed selection pipeline layout"),
        bind_group_layouts:   &[&motion_seed_selection_layout],
        push_constant_ranges: &[],
      });
    let motion_seed_selection_pipeline =
      device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:               Some("motion-line spacetime seed selection pipeline"),
        layout:              Some(&motion_seed_selection_pipeline_layout),
        module:              &motion_seed_selection_shader,
        entry_point:         Some("select"),
        compilation_options: Default::default(),
        cache:               None,
      });
    let motion_post_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("motion-line post-process pipeline layout"),
        bind_group_layouts:   &[&motion_post_layout],
        push_constant_ranges: &[],
      });
    let motion_post_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label:               Some("motion-line post-process pipeline"),
      layout:              Some(&motion_post_pipeline_layout),
      module:              &motion_post_shader,
      entry_point:         Some("postprocess"),
      compilation_options: Default::default(),
      cache:               None,
    });
    let motion_line_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("motion-line render pipeline layout"),
        bind_group_layouts:   &[&layout, &motion_line_layout],
        push_constant_ranges: &[],
      });
    let create_motion_line_pipeline = |label, vertex_entry, fragment_entry| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label:         Some(label),
      layout:        Some(&motion_line_pipeline_layout),
      vertex:        wgpu::VertexState {
        module:              &motion_line_shader,
        entry_point:         Some(vertex_entry),
        buffers:             &[],
        compilation_options: Default::default(),
      },
      fragment:      Some(wgpu::FragmentState {
        module:              &motion_line_shader,
        entry_point:         Some(fragment_entry),
        targets:             &[
          Some(wgpu::ColorTargetState {
            format:     wgpu::TextureFormat::Rgba16Float,
            // Weighted blended OIT accumulates premultiplied color and weight
            // independently of draw order.
            blend:      Some(wgpu::BlendState {
              color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation:  wgpu::BlendOperation::Add,
              },
              alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation:  wgpu::BlendOperation::Add,
              },
            }),
            write_mask: wgpu::ColorWrites::ALL,
          }),
          Some(wgpu::ColorTargetState {
            format:     wgpu::TextureFormat::R8Unorm,
            // Revealage starts at one and is multiplied by (1 - alpha).
            blend:      Some(wgpu::BlendState {
              color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation:  wgpu::BlendOperation::Add,
              },
              alpha: wgpu::BlendComponent::REPLACE,
            }),
            write_mask: wgpu::ColorWrites::RED,
          }),
        ],
        compilation_options: Default::default(),
      }),
      primitive:     wgpu::PrimitiveState {
        // Each trajectory is emitted as one view-aligned triangle strip with
        // two vertices per sample. Shared sample pairs connect every bend;
        // the fragment stage applies the depth-dependent halo displacement.
        topology: wgpu::PrimitiveTopology::TriangleStrip,
        ..Default::default()
      },
      depth_stencil: Some(wgpu::DepthStencilState {
        format:              wgpu::TextureFormat::Depth24Plus,
        // Keep the mesh depth test, but do not let one transparent line
        // update it before another line is accumulated. Otherwise the result
        // would still depend on the trajectory draw order.
        depth_write_enabled: false,
        depth_compare:       wgpu::CompareFunction::LessEqual,
        stencil:             Default::default(),
        bias:                Default::default(),
      }),
      multisample:   multisample_state(sample_count),
      multiview:     None,
      cache:         None,
    });
    let motion_line_pipeline = create_motion_line_pipeline("motion-line render pipeline", "line_vertex", "line_fragment");
    let motion_line_full_trajectory_pipeline = create_motion_line_pipeline(
      "motion-line full trajectory render pipeline",
      "full_trajectory_vertex",
      "full_trajectory_fragment",
    );
    let motion_line_windowed_full_trajectory_pipeline = create_motion_line_pipeline(
      "motion-line windowed full trajectory render pipeline",
      "full_trajectory_vertex",
      "windowed_full_trajectory_fragment",
    );
    let motion_line_unweighted_pipeline = create_motion_line_pipeline(
      "motion-line unweighted teaser render pipeline",
      "line_vertex",
      "unweighted_line_fragment",
    );
    let motion_line_dashed_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label:  Some("motion-line dashed render pipeline"),
      layout: Some(&motion_line_pipeline_layout),
      vertex: wgpu::VertexState {
        module:              &motion_line_dashed_shader,
        entry_point:         Some("line_vertex"),
        buffers:             &[],
        compilation_options: Default::default(),
      },
      fragment: Some(wgpu::FragmentState {
        module:      &motion_line_dashed_shader,
        entry_point: Some("dashed_line_fragment"),
        targets: &[
          Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState {
              color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
              },
              alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
              },
            }),
            write_mask: wgpu::ColorWrites::ALL,
          }),
          Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R8Unorm,
            blend: Some(wgpu::BlendState {
              color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
              },
              alpha: wgpu::BlendComponent::REPLACE,
            }),
            write_mask: wgpu::ColorWrites::RED,
          }),
        ],
        compilation_options: Default::default(),
      }),
      primitive: wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleStrip,
        ..Default::default()
      },
      depth_stencil: Some(wgpu::DepthStencilState {
        format:              wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: false,
        depth_compare:       wgpu::CompareFunction::LessEqual,
        stencil:             Default::default(),
        bias:                Default::default(),
      }),
      multisample: multisample_state(sample_count),
      multiview:   None,
      cache:       None,
    });
    let seed_point_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("seed-point layout"),
      entries: &[
        storage_layout_entry(0, wgpu::ShaderStages::VERTEX, true),
        storage_layout_entry(1, wgpu::ShaderStages::VERTEX, true),
        uniform_layout_entry(2, wgpu::ShaderStages::VERTEX),
      ],
    });
    let seed_point_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("seed-point pipeline layout"),
        bind_group_layouts: &[&layout, &skin_layout, &seed_point_layout],
        push_constant_ranges: &[],
      });
    let seed_point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("seed-point ring pipeline"),
      layout: Some(&seed_point_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &seed_point_shader,
        entry_point: Some("ring_vertex"),
        buffers: &[],
        compilation_options: Default::default(),
      },
      fragment: Some(wgpu::FragmentState {
        module: &seed_point_shader,
        entry_point: Some("ring_fragment"),
        targets: &[
          Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState {
              color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
              },
              alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
              },
            }),
            write_mask: wgpu::ColorWrites::ALL,
          }),
          Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::R8Unorm,
            blend: Some(wgpu::BlendState {
              color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
              },
              alpha: wgpu::BlendComponent::REPLACE,
            }),
            write_mask: wgpu::ColorWrites::RED,
          }),
        ],
        compilation_options: Default::default(),
      }),
      primitive: wgpu::PrimitiveState {
        cull_mode: None,
        ..Default::default()
      },
      depth_stencil: Some(wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::LessEqual,
        stencil: Default::default(),
        bias: Default::default(),
      }),
      multisample: multisample_state(sample_count),
      multiview: None,
      cache: None,
    });
    let motion_composite_pipeline_layout =
      device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("motion-line OIT composite pipeline layout"),
        bind_group_layouts:   &[&layout, &motion_composite_layout],
        push_constant_ranges: &[],
      });
    let motion_composite_pipeline =
      device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:         Some("motion-line OIT composite pipeline"),
        layout:        Some(&motion_composite_pipeline_layout),
        vertex:        wgpu::VertexState {
          module:              &motion_line_shader,
          entry_point:         Some("oit_composite_vertex"),
          buffers:             &[],
          compilation_options: Default::default(),
        },
        fragment:      Some(wgpu::FragmentState {
          module:              &motion_line_shader,
          entry_point:         Some("oit_composite_fragment"),
          targets:             &[Some(wgpu::ColorTargetState {
            format,
            // The resolved OIT color is composited over the mesh color that is
            // already present in the swap-chain texture.
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
          })],
          compilation_options: Default::default(),
        }),
        primitive:     wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample:   Default::default(),
        multiview:     None,
        cache:         None,
      });
    let motion_oit = MotionLineOitTargets::new(&device, &config, sample_count);
    let motion_composite_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label:   Some("motion-line OIT composite bind"),
      layout:  &motion_composite_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding:  0,
          resource: wgpu::BindingResource::TextureView(&motion_oit.accumulation_view),
        },
        wgpu::BindGroupEntry {
          binding:  1,
          resource: wgpu::BindingResource::TextureView(&motion_oit.revealage_view),
        },
      ],
    });
    // The mesh is retained on the CPU for camera framing, but it is not
    // uploaded here. The first queued action immediately knows whether it
    // is static or animated; uploading this bootstrap mesh would create a
    // second full geometry allocation during every animated startup.
    let placeholder = [Vertex {
      position: [0.0; 3],
      normal:   [0.0, 1.0, 0.0],
    }];
    let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label:    Some("static bootstrap vertex"),
      contents: bytemuck::cast_slice(&placeholder),
      usage:    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    });
    // Keep one valid index for the inactive static pipeline. Animated
    // scenes use their own AnimatedGpuState index buffer instead.
    let placeholder_indices = [0_u32, 0, 0];
    let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label:    Some("static bootstrap index"),
      contents: bytemuck::cast_slice(&placeholder_indices),
      usage:    wgpu::BufferUsages::INDEX,
    });
    let msaa_color = msaa_color_view(&device, &config, format, sample_count);
    let depth = depth_view(&device, &config, sample_count);
    let count = 0;
    let mut s = Self {
      camera,
      mesh,
      scene: None,
      animation_index: None,
      animation_time: 0.0,
      animation_speed: 1.0,
      animation_playing: false,
      scene_visible: true,
      camera_follow: None,
      camera_target_path: None,
      #[cfg(target_arch = "wasm32")]
      instance: _instance.expect("surface viewers retain their wgpu instance"),
      surface,
      #[cfg(not(target_arch = "wasm32"))]
      headless_target: _headless_target,
      device,
      queue,
      config,
      pipeline,
      animated_pipeline,
      phantom_pipeline,
      phantom_opacity_layout,
      motion_line_pipeline,
      motion_line_full_trajectory_pipeline,
      motion_line_windowed_full_trajectory_pipeline,
      motion_line_unweighted_pipeline,
      motion_line_dashed_pipeline,
      seed_point_pipeline,
      motion_composite_pipeline,
      motion_trace_pipeline,
      motion_post_pipeline,
      motion_seed_trace_pipeline,
      motion_seed_selection_pipeline,
      vertex,
      index,
      count,
      uniform,
      bind,
      skin_layout,
      motion_trace_layout,
      motion_post_layout,
      motion_seed_trace_layout,
      motion_seed_selection_layout,
      motion_line_layout,
      seed_point_layout,
      motion_composite_layout,
      motion_composite_bind,
      animated: None,
      phantoms: Vec::new(),
      motion_line_config: None,
      motion_line_style: MotionLineRenderStyle::default(),
      motion_line_opacity: 1.0,
      motion_lines_visible: true,
      seed_points_visible: false,
      motion_lines: None,
      last_motion_line_peak_bytes: 0,
      sample_count,
      msaa_color,
      depth,
      motion_oit,
      encode_srgb,
      // White is a neutral default for the presentation and for scripts
      // that do not need a custom backdrop.
      background_color: [1.0, 1.0, 1.0],
      #[cfg(not(target_arch = "wasm32"))]
      screenshot_path: None,
      #[cfg(not(target_arch = "wasm32"))]
      screenshot_complete: false,
    };
    s.reset_camera();
    Ok(s)
  }

  fn create_motion_composite_bind(&self) -> wgpu::BindGroup {
    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label:   Some("motion-line OIT composite bind"),
      layout:  &self.motion_composite_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding:  0,
          resource: wgpu::BindingResource::TextureView(&self.motion_oit.accumulation_view),
        },
        wgpu::BindGroupEntry {
          binding:  1,
          resource: wgpu::BindingResource::TextureView(&self.motion_oit.revealage_view),
        },
      ],
    })
  }

  /// Detaches the current canvas without destroying the device or pipelines.
  ///
  /// Slidev changes which canvas is visible as it changes slides. Keeping the
  /// device alive and dropping only this surface avoids creating one browser
  /// GPUDevice per slide while still releasing the old canvas attachment
  /// before its window is dropped.
  #[cfg(target_arch = "wasm32")]
  pub fn detach_surface(&mut self) {
    self.surface.take();
  }

  /// Attaches the existing renderer to a newly visible canvas window.
  ///
  /// The device, queue, shader modules, pipelines, and scene buffers remain
  /// shared. Only the swap-chain surface and size-dependent depth texture
  /// are recreated for the new DOM canvas.
  #[cfg(target_arch = "wasm32")]
  pub fn attach_window(&mut self, window: Arc<Window>) -> Result<()> {
    // Use the canvas backing store for the same reason as in `new`: the
    // winit ResizeObserver may not have delivered its first measurement
    // when this surface is attached to a newly visible slide.
    let size = Self::initial_surface_size(&window);
    let surface = self.instance.create_surface(window)?;
    self.config.width = size.width.max(1);
    self.config.height = size.height.max(1);
    self.camera.aspect = self.config.width as f32 / self.config.height as f32;
    surface.configure(&self.device, &self.config);
    self.msaa_color = msaa_color_view(
      &self.device,
      &self.config,
      self.config.format,
      self.sample_count,
    );
    self.depth = depth_view(&self.device, &self.config, self.sample_count);
    self.motion_oit.destroy();
    self.motion_oit = MotionLineOitTargets::new(&self.device, &self.config, self.sample_count);
    self.motion_composite_bind = self.create_motion_composite_bind();
    self.surface = Some(surface);
    Ok(())
  }

  /// Reconfigures the surface and recreates the depth texture after resize.
  pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
    if size.width > 0 && size.height > 0 {
      self.config.width = size.width;
      self.config.height = size.height;
      self.camera.aspect = size.width as f32 / size.height as f32;
      // Depth textures are size-dependent, so they must be recreated
      // together with the surface configuration.
      if let Some(surface) = &self.surface {
        surface.configure(&self.device, &self.config);
      }
      self.msaa_color = msaa_color_view(
        &self.device,
        &self.config,
        self.config.format,
        self.sample_count,
      );
      self.depth = depth_view(&self.device, &self.config, self.sample_count);
      self.motion_oit.destroy();
      self.motion_oit = MotionLineOitTargets::new(&self.device, &self.config, self.sample_count);
      self.motion_composite_bind = self.create_motion_composite_bind();
    }
  }

  /// Frames the current mesh using its precomputed bounding box.
  pub fn reset_camera(&mut self) {
    self.camera_follow = None;
    self.camera_target_path = None;
    self.camera.reset_for_bounds(self.mesh.min, self.mesh.max)
  }

  /// Frames the complete selected animation once and leaves the camera fixed.
  /// The sampled centroid path and per-pose radius are reused from the
  /// follow-camera preparation, but no follow state is installed.
  pub fn frame_animation(&mut self) {
    self.camera_follow = None;
    self.ensure_camera_target_path();
    let Some(path) = self.camera_target_path.as_ref() else {
      self.camera.reset_for_bounds(self.mesh.min, self.mesh.max);
      return;
    };
    if path.targets.is_empty() {
      self.camera.reset_for_bounds(self.mesh.min, self.mesh.max);
      return;
    }

    let mut min = path.targets[0];
    let mut max = path.targets[0];
    for &target in &path.targets[1..] {
      min = min.min(target);
      max = max.max(target);
    }
    let center = (min + max) * 0.5;
    let radius = path
      .targets
      .iter()
      .map(|&target| (target - center).length() + path.radius)
      .fold(path.radius, f32::max);
    // Keep the complete-clip framing while bringing the presentation view
    // closer to the fighter for the spacetime showcase slides.
    let extent = Vec3::splat((radius * 0.70).max(0.01));
    // Use a canonical presentation view instead of inheriting the previous
    // slide's camera direction or field of view.
    let direction = Vec3::new(1.0, 0.65, 1.0).normalize();
    self.camera.target = center;
    self.camera.eye = center + direction;
    self.camera.vertical_fov = 45.0;
    self
      .camera
      .reset_for_bounds(center - extent, center + extent);
  }

  /// Replaces the current static mesh and clears any glTF playback state.
  pub fn mesh_replace(&mut self, mesh: Mesh) {
    self.camera_follow = None;
    self.camera_target_path = None;
    self.scene = None;
    self.motion_line_config = None;
    self.motion_line_opacity = 1.0;
    self.phantoms.clear();
    self.clear_motion_lines();
    self.motion_lines_visible = true;
    self.seed_points_visible = false;
    self.clear_animated_gpu_state();
    self.animation_index = None;
    self.animation_time = 0.0;
    self.animation_playing = false;
    self.replace_mesh(mesh);
  }

  /// Replaces the current asset with a parsed glTF scene.
  pub fn scene_replace(&mut self, scene: AnimatedScene) {
    let mesh = scene.initial_mesh();
    self.scene_replace_with_mesh(scene, mesh);
  }

  /// Installs a parsed scene when its first frame was already sampled during
  /// viewer initialization. Keeping that frame avoids a duplicate CPU
  /// skinning pass for large animated FBX assets during startup.
  pub fn scene_replace_with_mesh(&mut self, scene: AnimatedScene, mesh: Mesh) {
    // Release the old scene before converting the new one. This avoids
    // retaining two complete CPU scene graphs while the next GPU mesh is
    // being prepared.
    self.motion_line_config = None;
    self.motion_line_opacity = 1.0;
    self.phantoms.clear();
    self.clear_motion_lines();
    self.motion_lines_visible = true;
    self.seed_points_visible = false;
    self.clear_animated_gpu_state();
    self.camera_follow = None;
    self.camera_target_path = None;
    self.scene = None;
    let gpu_mesh = scene.gpu_mesh();
    self.scene = Some(scene);
    self.animation_index = None;
    self.animation_time = 0.0;
    self.animation_playing = false;
    // Animated rendering uses AnimatedGpuState's vertex/index buffers.
    // Keep only a tiny valid static fallback instead of uploading the same
    // complete geometry a second time into Viewer::vertex/index.
    self.replace_static_placeholder(mesh);
    self.replace_animated_mesh(gpu_mesh);
  }

  /// Uploads the immutable animated geometry and its initial transform
  /// palette.  This is performed once when a scene is installed; subsequent
  /// frames update only `palette` with node/joint matrices.
  fn replace_animated_mesh(&mut self, mesh: AnimatedGpuMesh) {
    let vertex = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("animated vertices"),
        contents: bytemuck::cast_slice(&mesh.vertices),
        // The tracing compute shader reads the same packed vertices used by
        // the render pipeline, avoiding a second copy of the surface.
        usage:    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
      });
    let index = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("animated indices"),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage:    wgpu::BufferUsages::INDEX,
      });
    let palette = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("animated transform palette"),
        contents: bytemuck::cast_slice(&mesh.transforms),
        usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
      });
    // WebGPU rejects zero-sized buffers. A single dummy value keeps the
    // optional morph stream bindable for ordinary skeletal scenes.
    // Borrow the CPU vectors while uploading them instead of cloning the
    // complete morph streams. Large morph targets otherwise briefly exist
    // in CPU memory twice during scene replacement, which needlessly raises
    // the peak seen by the browser while GPU buffers are being allocated.
    let empty_morph_positions = [[0.0; 4]];
    let morph_positions_data = if mesh.morph_positions.is_empty() {
      &empty_morph_positions[..]
    } else {
      &mesh.morph_positions[..]
    };
    let empty_morph_weights = [0.0];
    let morph_weights_data = if mesh.morph_weights.is_empty() {
      &empty_morph_weights[..]
    } else {
      &mesh.morph_weights[..]
    };
    let morph_positions = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("animated morph positions"),
        contents: bytemuck::cast_slice(&morph_positions_data),
        usage:    wgpu::BufferUsages::STORAGE,
      });
    let morph_weights = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("animated morph weights"),
        contents: bytemuck::cast_slice(&morph_weights_data),
        usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
      });
    let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label:   Some("animated transform bind"),
      layout:  &self.skin_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding:  0,
          resource: palette.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  1,
          resource: morph_positions.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  2,
          resource: morph_weights.as_entire_binding(),
        },
      ],
    });
    self.animated = Some(AnimatedGpuState {
      vertex,
      index,
      palette,
      _morph_positions: morph_positions,
      morph_weights,
      bind,
      transforms: mesh.transforms,
      morph_weights_cpu: mesh.morph_weights,
      vertex_count: mesh.vertices.len(),
      count: mesh.indices.len() as u32,
    });
  }

  /// Rebuilds buffers for a new topology and reframes the camera once.
  fn replace_mesh(&mut self, mesh: Mesh) {
    // Release the previous topology before allocating the replacement.
    // This is especially important on the shared browser device, where a
    // new FBX can otherwise temporarily coexist with the previous FBX.
    self.vertex.destroy();
    self.index.destroy();
    self.vertex = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("vertices"),
        contents: bytemuck::cast_slice(&mesh.vertices),
        usage:    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
      });
    self.index = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("indices"),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage:    wgpu::BufferUsages::INDEX,
      });
    self.count = mesh.indices.len() as u32;
    self.mesh = mesh;
    self.reset_camera();
  }

  /// Replaces the ordinary static buffers with a tiny fallback for animated
  /// scenes. The static pipeline is not selected while `self.animated` is
  /// present, but the fields remain valid for a later static replacement.
  fn replace_static_placeholder(&mut self, mesh: Mesh) {
    self.vertex.destroy();
    self.index.destroy();
    let placeholder = [Vertex {
      position: [0.0; 3],
      normal:   [0.0, 1.0, 0.0],
    }];
    let placeholder_indices = [0_u32, 0, 0];
    self.vertex = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("static fallback vertex"),
        contents: bytemuck::cast_slice(&placeholder),
        usage:    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
      });
    self.index = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("static fallback index"),
        contents: bytemuck::cast_slice(&placeholder_indices),
        usage:    wgpu::BufferUsages::INDEX,
      });
    self.count = 0;
    self.mesh = mesh;
    self.reset_camera();
  }

  /// Drops the animated GPU state and explicitly destroys its buffers.
  fn clear_animated_gpu_state(&mut self) {
    if let Some(animated) = self.animated.take() {
      animated.destroy();
    }
  }

  /// Drops the currently extracted trajectory bundle and its compute inputs.
  pub fn clear_motion_lines(&mut self) {
    self.motion_line_config = None;
    if let Some(lines) = self.motion_lines.take() {
      lines.destroy();
    }
  }

  /// Changes only the rendering style of the current/future line bundle.
  pub fn set_motion_line_style(&mut self, style: MotionLineRenderStyle) {
    self.motion_line_style = style;
  }

  /// Sets the opacity used by full-trajectory styles for this viewer only.
  pub fn set_motion_line_opacity(&mut self, opacity: f32) {
    self.motion_line_opacity = opacity.clamp(0.0, 1.0);
  }

  /// Shows the selected motion-line vertices as rings at their live pose.
  pub fn set_seed_points_visible(&mut self, visible: bool) {
    self.seed_points_visible = visible;
  }

  /// Controls drawing without discarding the selected seeds or trajectories.
  pub fn set_motion_lines_visible(&mut self, visible: bool) {
    self.motion_lines_visible = visible;
  }

  /// Prints process RSS and the exact sizes of buffers owned by this viewer.
  ///
  /// wgpu deliberately does not expose portable live VRAM counters. The GPU
  /// figure is therefore the useful lower bound of resources allocated by the
  /// viewer (including the depth texture estimate), while CPU RSS is read from
  /// the native process where available. Lua examples call this after queued
  /// setup actions so the report describes the actual ready viewer.
  pub fn print_memory_usage(&self, label: &str) {
    let static_gpu_bytes = self.vertex.size() + self.index.size() + self.uniform.size();
    let animated_gpu_bytes = self
      .animated
      .as_ref()
      .map(|animated| {
        animated.vertex.size()
          + animated.index.size()
          + animated.palette.size()
          + animated._morph_positions.size()
          + animated.morph_weights.size()
      })
      .unwrap_or(0);
    let motion_gpu_bytes = self
      .motion_lines
      .as_ref()
      .map(MotionLineGpuState::gpu_buffer_bytes)
      .unwrap_or(0);
    let motion_oit_bytes = self.motion_oit.gpu_bytes();
    let pixels = self.config.width as u64 * self.config.height as u64;
    let depth_bytes = pixels * 4 * self.sample_count as u64;
    // Surface formats used by the presentation path are normally four bytes
    // per pixel. This estimate includes the private multisampled color target
    // but intentionally does not pretend to know the driver's swap-chain size.
    let msaa_color_bytes = if self.sample_count > 1 {
      pixels * 4 * self.sample_count as u64
    } else {
      0
    };
    let tracked_gpu_bytes = static_gpu_bytes
      + animated_gpu_bytes
      + motion_gpu_bytes
      + motion_oit_bytes
      + depth_bytes
      + msaa_color_bytes;
    let cpu = process_rss_bytes()
      .map(format_bytes)
      .unwrap_or_else(|| "unavailable on this platform".to_owned());
    eprintln!(
      "[memory] {label}: CPU RSS {cpu}; GPU tracked {} (scene {}, motion lines {}, OIT {}, MSAA color {}, depth {}, {}x); last motion-line extraction peak {}",
      format_bytes(tracked_gpu_bytes),
      format_bytes(static_gpu_bytes + animated_gpu_bytes),
      format_bytes(motion_gpu_bytes),
      format_bytes(motion_oit_bytes),
      format_bytes(msaa_color_bytes),
      format_bytes(depth_bytes),
      self.sample_count,
      format_bytes(self.last_motion_line_peak_bytes),
    );
  }

  /// Returns all animations in the currently loaded glTF scene.
  pub fn animation_infos(&self) -> Vec<AnimationInfo> {
    self
      .scene
      .as_ref()
      .map(AnimatedScene::animations)
      .unwrap_or_default()
  }

  /// Selects an animation and shows its first frame without starting it.
  pub fn select_animation(&mut self, index: usize) {
    let Some(scene) = self.scene.as_ref() else {
      return;
    };
    if index >= scene.animations().len() {
      return;
    }
    self.animation_index = Some(index);
    self.animation_time = 0.0;
    self.animation_playing = false;
    self.camera_target_path = None;
    self.upload_animation_pose(Some(index), 0.0);
    if self.motion_line_config.is_some() {
      if let Err(error) = self.rebuild_motion_lines() {
        eprintln!("could not rebuild motion lines: {error:#}");
      }
    }
  }

  /// Starts the selected animation, defaulting to the first animation.
  pub fn play_animation(&mut self) {
    if self.animation_index.is_none() && !self.animation_infos().is_empty() {
      self.animation_index = Some(0);
      self.animation_time = 0.0;
      self.upload_animation_pose(Some(0), 0.0);
      if self.motion_line_config.is_some() {
        if let Err(error) = self.rebuild_motion_lines() {
          eprintln!("could not rebuild motion lines: {error:#}");
        }
      }
    }
    self.animation_playing = self.animation_index.is_some();
  }

  /// Pauses playback while retaining the selected frame and time.
  pub fn pause_animation(&mut self) {
    self.animation_playing = false;
  }

  /// Sets playback speed. Negative values play backwards; zero pauses time
  /// progression without changing the selected playing state.
  pub fn set_animation_speed(&mut self, speed: f32) {
    self.animation_speed = speed;
  }

  /// Selects seeds, prepares uniform animation samples, and launches the GPU
  /// trace plus adaptive Catmull–Rom post-process passes.
  pub fn configure_motion_lines(&mut self, config: MotionLineConfig) -> Result<()> {
    config.validate()?;
    let previous = self.motion_line_config.replace(config);
    if let Err(error) = self.rebuild_motion_lines() {
      self.motion_line_config = previous;
      return Err(error);
    }
    Ok(())
  }


  /// Traces every animated vertex at the low rate requested by spacetime
  /// seeding and selects the max-min seed set on the GPU. Only the resulting
  /// seed-index buffer survives this method; the all-vertex position stream,
  /// low-rate palettes, and temporary bind groups are destroyed immediately
  /// after their command submission.
  fn trace_spacetime_seed_selection(
    &mut self,
    pose: &PoseSamples,
    vertex_count: usize,
    seed_count: u32,
    sampling_rate: f32,
    options: SpacetimeSelectionOptions,
  ) -> Result<wgpu::Buffer> {
    if vertex_count == 0 || seed_count == 0 {
      bail!("uniform spacetime selection requires at least one source vertex")
    }
    let vertex_count_u32 = u32::try_from(vertex_count)
      .map_err(|_| anyhow!("spacetime seed vertex count exceeds the GPU addressable range"))?;
    let max_storage = self.device.limits().max_storage_buffer_binding_size as u64;
    let position_count = (vertex_count_u32 as u64)
      .checked_mul(pose.sample_count as u64)
      .ok_or_else(|| anyhow!("spacetime seed position count overflow"))?;
    let position_bytes = position_count
      .checked_mul(std::mem::size_of::<[f32; 4]>() as u64)
      .ok_or_else(|| anyhow!("spacetime seed position buffer size overflow"))?;
    let position_budget = MAX_SPACETIME_SEED_POSITION_BYTES.min(max_storage);
    if position_bytes > position_budget {
      bail!(
        "uniform spacetime seed positions need {position_bytes} bytes, but the temporary budget is {position_budget} bytes; lower the seed sampling rate"
      )
    }
    let palette_bytes = (pose.palettes.len() as u64)
      .checked_mul(std::mem::size_of::<SkinTransform>() as u64)
      .ok_or_else(|| anyhow!("spacetime seed palette size overflow"))?;
    if palette_bytes > max_storage {
      bail!(
        "uniform spacetime seed palettes need {palette_bytes} bytes, but this GPU supports only {max_storage} bytes per storage binding"
      )
    }
    let morph_bytes = (pose.morph_weights.len() as u64)
      .checked_mul(std::mem::size_of::<f32>() as u64)
      .ok_or_else(|| anyhow!("spacetime seed morph-weight size overflow"))?;
    if morph_bytes > max_storage {
      bail!(
        "uniform spacetime seed morph weights need {morph_bytes} bytes, but this GPU supports only {max_storage} bytes per storage binding"
      )
    }
    let seed_bytes = (seed_count as u64)
      .checked_mul(std::mem::size_of::<u32>() as u64)
      .ok_or_else(|| anyhow!("spacetime seed output size overflow"))?;
    if seed_bytes > max_storage {
      bail!(
        "uniform spacetime seed output needs {seed_bytes} bytes, but this GPU supports only {max_storage} bytes per storage binding"
      )
    }
    let pair_bytes = (vertex_count_u32 as u64)
      .checked_mul(std::mem::size_of::<[f32; 4]>() as u64)
      .ok_or_else(|| anyhow!("spacetime seed pair scratch size overflow"))?;
    let candidate_nearest_bytes = position_count
      .checked_mul(std::mem::size_of::<f32>() as u64)
      .ok_or_else(|| anyhow!("spacetime seed nearest-distance scratch size overflow"))?;
    let candidate_score_bytes = (vertex_count_u32 as u64)
      .checked_mul(std::mem::size_of::<f32>() as u64)
      .ok_or_else(|| anyhow!("spacetime seed score scratch size overflow"))?;
    for (label, bytes) in [
      ("pair scratch", pair_bytes),
      ("nearest-distance scratch", candidate_nearest_bytes),
      ("candidate score scratch", candidate_score_bytes),
    ] {
      if bytes > max_storage {
        bail!(
          "uniform spacetime seed {label} needs {bytes} bytes, but this GPU supports only {max_storage} bytes per storage binding"
        )
      }
    }
    let selection_pass_count = if seed_count <= 2 {
      seed_count as u64
    } else {
      2 * seed_count as u64 - 2
    };
    if selection_pass_count > MAX_SPACETIME_SELECTION_PASSES {
      bail!(
        "uniform spacetime selection needs {selection_pass_count} dependent GPU passes, but the safety limit is {MAX_SPACETIME_SELECTION_PASSES}; lower the seed count"
      )
    }
    // All selector passes reuse one tiny uniform buffer. They are submitted
    // separately because every pass depends on the previous pass's result;
    // queue ordering makes each write visible to exactly the following pass.
    let selection_params_bytes = std::mem::size_of::<SpacetimeSeedSelectionParams>() as u64;
    let temporary_bytes = position_bytes
      .checked_add(palette_bytes)
      .and_then(|bytes| bytes.checked_add(morph_bytes))
      .and_then(|bytes| bytes.checked_add(seed_bytes))
      .and_then(|bytes| bytes.checked_add(pair_bytes.max(16)))
      .and_then(|bytes| bytes.checked_add(candidate_nearest_bytes.max(4)))
      .and_then(|bytes| bytes.checked_add(candidate_score_bytes.max(4)))
      .and_then(|bytes| bytes.checked_add(std::mem::size_of::<SpacetimeSeedTraceParams>() as u64))
      .and_then(|bytes| bytes.checked_add(selection_params_bytes))
      .ok_or_else(|| anyhow!("spacetime seed temporary memory size overflow"))?;
    let previous_motion_bytes = self
      .motion_lines
      .as_ref()
      .map(MotionLineGpuState::gpu_buffer_bytes)
      .unwrap_or(0);
    if previous_motion_bytes
      .checked_add(temporary_bytes)
      .is_none_or(|bytes| bytes > MAX_MOTION_LINE_PEAK_BYTES)
    {
      bail!(
        "uniform spacetime seed extraction needs {temporary_bytes} temporary bytes while the previous bundle uses {previous_motion_bytes}; the peak budget is {MAX_MOTION_LINE_PEAK_BYTES} bytes"
      )
    }

    let animated = self
      .animated
      .as_ref()
      .ok_or_else(|| anyhow!("animated GPU geometry is not ready"))?;
    let positions_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("temporary spacetime seed positions"),
      size:               position_bytes.max(16),
      usage:              wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false,
    });
    let palettes_buffer = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("temporary spacetime seed palettes"),
        contents: bytemuck::cast_slice(&pose.palettes),
        usage:    wgpu::BufferUsages::STORAGE,
      });
    let morph_weights_buffer = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("temporary spacetime seed morph weights"),
        contents: bytemuck::cast_slice(&pose.morph_weights),
        usage:    wgpu::BufferUsages::STORAGE,
      });
    let trace_params = SpacetimeSeedTraceParams {
      vertex_count:        vertex_count_u32,
      sample_count:        pose.sample_count,
      vertex_word_stride:  (std::mem::size_of::<SkinnedVertex>() / 4) as u32,
      palette_stride:      pose.palette_stride,
      morph_weight_stride: pose.morph_stride,
      samples_per_second:  sampling_rate,
      duration:            pose.duration,
      _padding:            [0; 1],
    };
    let trace_params_buffer = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("spacetime seed trace parameters"),
        contents: bytemuck::bytes_of(&trace_params),
        usage:    wgpu::BufferUsages::UNIFORM,
      });
    let seeds_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("spacetime selected motion-line seeds"),
      size:               seed_bytes.max(4),
      usage:              wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false,
    });
    let pair_scores_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("spacetime seed pair scores"),
      size:               pair_bytes.max(16),
      usage:              wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false,
    });
    let candidate_nearest_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("spacetime seed candidate nearest distances"),
      size:               candidate_nearest_bytes.max(4),
      usage:              wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false,
    });
    let candidate_scores_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("spacetime seed candidate scores"),
      size:               candidate_score_bytes.max(4),
      usage:              wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false,
    });
    let trace_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label:   Some("spacetime seed trace bind"),
      layout:  &self.motion_seed_trace_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding:  0,
          resource: animated.vertex.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  1,
          resource: palettes_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  2,
          resource: animated._morph_positions.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  3,
          resource: morph_weights_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  4,
          resource: positions_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  5,
          resource: trace_params_buffer.as_entire_binding(),
        },
      ],
    });
    let selection_workgroups = (vertex_count_u32 + 63) / 64;
    let max_workgroups = self.device.limits().max_compute_workgroups_per_dimension;
    let pair_workgroups_x = vertex_count_u32.min(max_workgroups);
    let pair_workgroups_y = vertex_count_u32
      .checked_add(pair_workgroups_x - 1)
      .map(|count| count / pair_workgroups_x)
      .ok_or_else(|| anyhow!("spacetime seed pair dispatch size overflow"))?;
    if pair_workgroups_y > max_workgroups {
      bail!("spacetime seed pair dispatch exceeds the GPU workgroup limit")
    }
    let selection_params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("spacetime seed selection parameters"),
      size:               selection_params_bytes,
      usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    let selection_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label:   Some("spacetime seed selection bind"),
      layout:  &self.motion_seed_selection_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding:  0,
          resource: positions_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  1,
          resource: pair_scores_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  2,
          resource: candidate_nearest_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  3,
          resource: candidate_scores_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  4,
          resource: seeds_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  5,
          resource: selection_params_buffer.as_entire_binding(),
        },
      ],
    });
    let mut selection_passes = Vec::with_capacity(selection_pass_count as usize);
    let mut add_selection_pass =
      |stage: u32, selected_count: u32, workgroups_x: u32, workgroups_y: u32| {
        selection_passes.push((stage, selected_count, workgroups_x, workgroups_y));
    };
    if seed_count > 1 {
      // Stage 0 uses selected_count as the x dimension when mapping its
      // two-dimensional dispatch back to one left-hand vertex per group.
      add_selection_pass(0, pair_workgroups_x, pair_workgroups_x, pair_workgroups_y);
    }
    // This pass also handles the one-seed degenerate case.
    add_selection_pass(1, 0, 1, 1);
    if seed_count > 2 {
      add_selection_pass(2, 2, selection_workgroups, 1);
      for selected_count in 2..seed_count {
        add_selection_pass(3, selected_count, 1, 1);
        if selected_count + 1 < seed_count {
          add_selection_pass(4, selected_count, selection_workgroups, 1);
        }
      }
    }
    drop(add_selection_pass);
    let mut encoder = self
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("spacetime seed trace"),
      });
    {
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label:            Some("spacetime seed trace pass"),
        timestamp_writes: None,
      });
      pass.set_pipeline(&self.motion_seed_trace_pipeline);
      pass.set_bind_group(0, &trace_bind, &[]);
      pass.dispatch_workgroups((vertex_count_u32 + 7) / 8, (pose.sample_count + 7) / 8, 1);
    }
    self.queue.submit(Some(encoder.finish()));

    for (stage, selected_count, workgroups_x, workgroups_y) in selection_passes {
      let selection_params = SpacetimeSeedSelectionParams {
        vertex_count: vertex_count_u32,
        sample_count: pose.sample_count,
        seed_count,
        stage,
        selected_count,
        importance: u32::from(options.importance),
        stochastic: u32::from(options.stochastic),
        extended: u32::from(options.extended),
      };
      self.queue.write_buffer(
        &selection_params_buffer,
        0,
        bytemuck::bytes_of(&selection_params),
      );
      let mut encoder = self
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
          label: Some("spacetime seed selection"),
        });
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label:            Some("spacetime seed selection pass"),
        timestamp_writes: None,
      });
      pass.set_pipeline(&self.motion_seed_selection_pipeline);
      pass.set_bind_group(0, &selection_bind, &[]);
      pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
      drop(pass);
      self.queue.submit(Some(encoder.finish()));
    }

    drop(trace_bind);
    drop(selection_bind);
    selection_params_buffer.destroy();
    positions_buffer.destroy();
    palettes_buffer.destroy();
    morph_weights_buffer.destroy();
    trace_params_buffer.destroy();
    pair_scores_buffer.destroy();
    candidate_nearest_buffer.destroy();
    candidate_scores_buffer.destroy();
    Ok(seeds_buffer)
  }

  /// Prepares the sample-major palette stream, traces every selected
  /// seed/sample pair, and then adaptively resamples each trajectory. Both
  /// compute passes stay on the GPU; only the compact setup streams cross the
  /// CPU/GPU boundary.
  fn rebuild_motion_lines(&mut self) -> Result<()> {
    let config = self
      .motion_line_config
      .as_ref()
      .ok_or_else(|| anyhow!("motion-line configuration is not set"))?
      .clone();
    let animation = self
      .animation_index
      .ok_or_else(|| anyhow!("select an animation before configuring motion lines"))?;
    let (vertex_count, palette_stride, morph_stride) = self
      .animated
      .as_ref()
      .ok_or_else(|| anyhow!("animated GPU geometry is not ready"))
      .map(|animated| {
        (
          animated.vertex_count,
          animated.transforms.len(),
          animated.morph_weights_cpu.len(),
        )
      })?;
    if self.mesh.vertices.len() != vertex_count {
      bail!("motion-line source and animated vertex counts differ")
    }
    let (samples, spacetime_seed_buffer) = match config.seed_selection.spacetime_options() {
      Some((count, sampling_rate, options)) => {
        let seed_count = count.min(vertex_count);
        let low_rate_pose = {
          let scene = self
            .scene
            .as_ref()
            .ok_or_else(|| anyhow!("motion lines require an animated scene"))?;
          prepare_pose_samples(
            scene,
            animation,
            palette_stride,
            morph_stride,
            sampling_rate,
          )?
        };
        let seed_buffer = self.trace_spacetime_seed_selection(
          &low_rate_pose,
          vertex_count,
          seed_count as u32,
          sampling_rate,
          options,
        )?;
        // The low-rate CPU pose vectors are no longer needed once the GPU
        // selector has been submitted. Do not retain them alongside the
        // higher-rate final motion-line pose stream.
        drop(low_rate_pose);
        let final_pose = {
          let scene = self
            .scene
            .as_ref()
            .ok_or_else(|| anyhow!("motion lines require an animated scene"))?;
          prepare_pose_samples(
            scene,
            animation,
            palette_stride,
            morph_stride,
            config.frames_per_second,
          )?
        };
        (
          crate::motion_lines::MotionLineSamples {
            // The GPU-selected seed buffer is used below. This compact CPU
            // placeholder carries only the count through existing sizing
            // logic and is never uploaded.
            seeds:          vec![0; seed_count],
            palettes:       final_pose.palettes,
            morph_weights:  final_pose.morph_weights,
            duration:       final_pose.duration,
            sample_count:   final_pose.sample_count,
            palette_stride: final_pose.palette_stride,
            morph_stride:   final_pose.morph_stride,
          },
          Some(seed_buffer),
        )
      }
      _ => {
        // The initial sampled mesh has the same vertex order as the immutable
        // animated GPU mesh. Borrowing its positions lets static uniform
        // selection inspect the surface without another position array.
        let scene = self
          .scene
          .as_ref()
          .ok_or_else(|| anyhow!("motion lines require an animated scene"))?;
        let source_positions = &self.mesh.vertices;
        (
          prepare_samples(
            scene,
            animation,
            vertex_count,
            |index| source_positions[index].position,
            palette_stride,
            morph_stride,
            &config,
          )?,
          None,
        )
      }
    };
    let seed_count = u32::try_from(samples.seeds.len())
      .map_err(|_| anyhow!("motion-line seed count exceeds the GPU addressable range"))?;
    let max_storage = self.device.limits().max_storage_buffer_binding_size as u64;
    let trajectory_sample_bytes = std::mem::size_of::<TrajectorySample>() as u64;
    let output_budget = MAX_MOTION_LINE_POSTPROCESS_BYTES.min(max_storage);
    let bytes_per_seed_at_minimum = (seed_count as u64)
      .checked_mul(trajectory_sample_bytes)
      .ok_or_else(|| anyhow!("motion-line output size overflow"))?;
    let max_output_samples_per_seed = output_budget / bytes_per_seed_at_minimum.max(1);
    if max_output_samples_per_seed < samples.sample_count as u64 {
      let minimum_output_bytes = (seed_count as u64)
        .checked_mul(samples.sample_count as u64)
        .and_then(|count| count.checked_mul(trajectory_sample_bytes))
        .ok_or_else(|| anyhow!("motion-line minimum output size overflow"))?;
      bail!(
        "motion-line uniform samples need {minimum_output_bytes} bytes, but the post-process budget is {output_budget} bytes; lower the seed count or FPS"
      )
    }
    // Pick the largest power-of-two subdivision level that fits the output
    // budget. The shader always resamples each interval at least twice, so
    // fail explicitly if the memory budget cannot hold a finer curve.
    let segment_count = (samples.sample_count as u64).saturating_sub(1);
    let max_steps_per_segment = if segment_count == 0 {
      1
    } else {
      ((max_output_samples_per_seed.saturating_sub(1) / segment_count).max(1)) as u32
    };
    let mut max_subdivisions = 1_u32;
    while max_subdivisions < MAX_MOTION_LINE_SUBDIVISIONS
      && (max_subdivisions as u64) * 2 <= max_steps_per_segment as u64
    {
      max_subdivisions *= 2;
    }
    if samples.sample_count > 1 && max_subdivisions < 2 {
      bail!(
        "motion-line post-processing needs at least two spline steps per sampled interval; lower the seed count or FPS"
      )
    }
    let output_stride = (samples.sample_count as u64)
      .saturating_sub(1)
      .checked_mul(max_subdivisions as u64)
      .and_then(|count| count.checked_add(1))
      .and_then(|count| u32::try_from(count).ok())
      .ok_or_else(|| anyhow!("motion-line post-process output size overflow"))?;
    let model_diagonal = (self.mesh.max - self.mesh.min).length();
    // Compare the spline against its chords at a scale smaller than the
    // visible ribbon width, so bends receive extra samples where needed.
    let tolerance = (model_diagonal * 0.0005).max(1.0e-5);
    let params = MotionLineParams {
      seed_count,
      sample_count: samples.sample_count,
      output_stride,
      vertex_word_stride: (std::mem::size_of::<SkinnedVertex>() / 4) as u32,
      palette_stride: samples.palette_stride,
      morph_weight_stride: samples.morph_stride,
      max_subdivisions,
      samples_per_second: config.frames_per_second,
      duration: samples.duration,
      tolerance,
      reserved: [0.0; 3],
      _padding: [0; 3],
    };
    let raw_trajectory_count = (params.seed_count as u64)
      .checked_mul(params.sample_count as u64)
      .ok_or_else(|| anyhow!("motion-line trajectory size overflow"))?;
    let output_trajectory_count = (params.seed_count as u64)
      .checked_mul(params.output_stride as u64)
      .ok_or_else(|| anyhow!("motion-line post-process output size overflow"))?;
    let raw_trajectory_bytes = raw_trajectory_count
      .checked_mul(std::mem::size_of::<TrajectorySample>() as u64)
      .ok_or_else(|| anyhow!("motion-line raw trajectory byte size overflow"))?;
    let output_trajectory_bytes = output_trajectory_count
      .checked_mul(std::mem::size_of::<TrajectorySample>() as u64)
      .ok_or_else(|| anyhow!("motion-line trajectory byte size overflow"))?;
    let count_bytes = (params.seed_count as u64)
      .checked_mul(std::mem::size_of::<u32>() as u64)
      .ok_or_else(|| anyhow!("motion-line count byte size overflow"))?;
    if raw_trajectory_bytes > max_storage {
      bail!(
        "motion-line raw trajectory needs {raw_trajectory_bytes} bytes, but this GPU supports only {max_storage} bytes per storage binding"
      )
    }
    if output_trajectory_bytes > max_storage {
      bail!(
        "motion-line post-process output needs {output_trajectory_bytes} bytes, but this GPU supports only {max_storage} bytes per storage binding"
      )
    }
    if count_bytes > max_storage {
      bail!(
        "motion-line sample counts need {count_bytes} bytes, but this GPU supports only {max_storage} bytes per storage binding"
      )
    }
    let palette_bytes = (samples.palettes.len() as u64)
      .checked_mul(std::mem::size_of::<SkinTransform>() as u64)
      .ok_or_else(|| anyhow!("motion-line palette byte size overflow"))?;
    if palette_bytes > max_storage {
      bail!(
        "motion-line sample palettes need {palette_bytes} bytes, but this GPU supports only {max_storage} bytes per storage binding"
      )
    }
    let morph_bytes = (samples.morph_weights.len() as u64)
      .checked_mul(std::mem::size_of::<f32>() as u64)
      .ok_or_else(|| anyhow!("motion-line morph-weight byte size overflow"))?;
    if morph_bytes > max_storage {
      bail!(
        "motion-line sample morph weights need {morph_bytes} bytes, but this GPU supports only {max_storage} bytes per storage binding"
      )
    }
    let seeds_bytes = (params.seed_count as u64)
      .checked_mul(std::mem::size_of::<u32>() as u64)
      .ok_or_else(|| anyhow!("motion-line seed byte size overflow"))?;
    let style_bytes = std::mem::size_of::<MotionLineStyle>() as u64;
    let new_motion_peak_bytes = raw_trajectory_bytes
      .checked_add(output_trajectory_bytes)
      .and_then(|bytes| bytes.checked_add(palette_bytes))
      .and_then(|bytes| bytes.checked_add(morph_bytes))
      .and_then(|bytes| bytes.checked_add(seeds_bytes))
      .and_then(|bytes| bytes.checked_add(count_bytes))
      .and_then(|bytes| bytes.checked_add(std::mem::size_of::<MotionLineParams>() as u64))
      .and_then(|bytes| bytes.checked_add(style_bytes))
      .ok_or_else(|| anyhow!("motion-line peak memory size overflow"))?;
    let previous_motion_bytes = self
      .motion_lines
      .as_ref()
      .map(MotionLineGpuState::gpu_buffer_bytes)
      .unwrap_or(0);
    let motion_peak_bytes = previous_motion_bytes
      .checked_add(new_motion_peak_bytes)
      .ok_or_else(|| anyhow!("motion-line peak memory size overflow"))?;
    if motion_peak_bytes > MAX_MOTION_LINE_PEAK_BYTES {
      bail!(
        "motion-line extraction needs {new_motion_peak_bytes} bytes while the previous bundle uses {previous_motion_bytes}; the peak budget is {MAX_MOTION_LINE_PEAK_BYTES} bytes"
      )
    }
    // Keep the old bundle alive until the replacement has been fully created.
    // If validation or allocation fails, the previous visible bundle remains
    // usable instead of leaving the viewer in a half-configured state.
    let seeds_buffer = spacetime_seed_buffer.unwrap_or_else(|| {
      self
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
          label:    Some("motion-line seeds"),
          contents: bytemuck::cast_slice(&samples.seeds),
          usage:    wgpu::BufferUsages::STORAGE,
        })
    });
    let palettes_buffer = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("motion-line sample palettes"),
        contents: bytemuck::cast_slice(&samples.palettes),
        usage:    wgpu::BufferUsages::STORAGE,
      });
    let morph_weights_buffer = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("motion-line sample morph weights"),
        contents: bytemuck::cast_slice(&samples.morph_weights),
        usage:    wgpu::BufferUsages::STORAGE,
      });
    let raw_trajectories_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("motion-line uniformly sampled trajectories"),
      size:               raw_trajectory_bytes.max(16),
      usage:              wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false,
    });
    let trajectories_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("motion-line adaptively sampled trajectories"),
      size:               output_trajectory_bytes.max(16),
      usage:              wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false,
    });
    let counts_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
      label:              Some("motion-line trajectory sample counts"),
      size:               count_bytes.max(4),
      usage:              wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false,
    });
    let params_buffer = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("motion-line parameters"),
        contents: bytemuck::bytes_of(&params),
        usage:    wgpu::BufferUsages::UNIFORM,
      });
    let style = MotionLineStyle {
      timing:   [0.0, 0.75, model_diagonal.max(1.0e-4), 0.0],
      // Initial state is the unchanged teaser style. The per-frame write
      // below switches these values when the script selected Compasso.
      widths:   [34.0, 0.01, 0.0, 0.0],
      viewport: [
        self.config.width as f32,
        self.config.height as f32,
        0.0,
        0.0,
      ],
    };
    let style_buffer = self
      .device
      .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("motion-line style"),
        contents: bytemuck::bytes_of(&style),
        usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      });
    let animated = self
      .animated
      .as_ref()
      .ok_or_else(|| anyhow!("animated GPU geometry is not ready"))?;
    let trace_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label:   Some("motion-line compute bind"),
      layout:  &self.motion_trace_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding:  0,
          resource: animated.vertex.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  1,
          resource: palettes_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  2,
          resource: animated._morph_positions.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  3,
          resource: morph_weights_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  4,
          resource: seeds_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  5,
          resource: raw_trajectories_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  6,
          resource: params_buffer.as_entire_binding(),
        },
      ],
    });
    let post_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label:   Some("motion-line post-process bind"),
      layout:  &self.motion_post_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding:  0,
          resource: raw_trajectories_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  1,
          resource: trajectories_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  2,
          resource: counts_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  3,
          resource: params_buffer.as_entire_binding(),
        },
      ],
    });
    let line_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label:   Some("motion-line render bind"),
      layout:  &self.motion_line_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding:  0,
          resource: trajectories_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  1,
          resource: params_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  2,
          resource: counts_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding:  3,
          resource: style_buffer.as_entire_binding(),
        },
      ],
    });
    let seed_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("seed-point render bind"),
      layout: &self.seed_point_layout,
      entries: &[
        wgpu::BindGroupEntry {
          binding: 0,
          resource: animated.vertex.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: seeds_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: style_buffer.as_entire_binding(),
        },
      ],
    });
    let mut encoder = self
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("motion-line trace and post-process"),
      });
    {
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label:            Some("motion-line trace pass"),
        timestamp_writes: None,
      });
      pass.set_pipeline(&self.motion_trace_pipeline);
      pass.set_bind_group(0, &trace_bind, &[]);
      pass.dispatch_workgroups(
        (params.seed_count + 7) / 8,
        (params.sample_count + 7) / 8,
        1,
      );
    }
    {
      let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label:            Some("motion-line post-process pass"),
        timestamp_writes: None,
      });
      pass.set_pipeline(&self.motion_post_pipeline);
      pass.set_bind_group(0, &post_bind, &[]);
      pass.dispatch_workgroups((params.seed_count + 63) / 64, 1, 1);
    }
    self.queue.submit(Some(encoder.finish()));
    self.last_motion_line_peak_bytes = motion_peak_bytes;
    // These inputs are consumed only by the submitted compute passes. wgpu
    // keeps the submitted work valid after destroy(), but releasing the
    // handles here prevents the large sampled palettes/raw bundle from
    // remaining resident for the lifetime of the rendered line bundle.
    drop(trace_bind);
    drop(post_bind);
    palettes_buffer.destroy();
    morph_weights_buffer.destroy();
    raw_trajectories_buffer.destroy();
    let next = MotionLineGpuState {
      trajectories: trajectories_buffer,
      counts: counts_buffer,
      params: params_buffer,
      style: style_buffer,
      seeds: seeds_buffer,
      line_bind,
      seed_bind,
      seed_count: params.seed_count,
      output_stride: params.output_stride,
    };
    if let Some(previous) = self.motion_lines.replace(next) {
      previous.destroy();
    }
    Ok(())
  }

  /// Jumps to a time in the selected animation.
  pub fn set_animation_time(&mut self, time: f32) {
    let Some(index) = self.animation_index else {
      return;
    };
    self.animation_time = time;
    self.upload_animation_pose(Some(index), time);
    self.update_follow_camera();
  }

  /// Adds a translucent GPU-skinned copy of the selected animation at `time`.
  /// Each phantom owns its sampled palette while sharing the source geometry.
  pub fn render_phantom(&mut self, time: f32, opacity: f32) {
    let Some(index) = self.animation_index else { return };
    let Some(scene) = self.scene.as_ref() else { return };
    let Some(animated) = self.animated.as_ref() else { return };

    let mut transforms = Vec::with_capacity(animated.transforms.len());
    let mut morph_weights = Vec::with_capacity(animated.morph_weights_cpu.len());
    scene.update_gpu_pose(Some(index), time, &mut transforms, &mut morph_weights);
    let palette = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("phantom transform palette"),
      contents: bytemuck::cast_slice(&transforms),
      usage: wgpu::BufferUsages::STORAGE,
    });
    let zero_weight = [0.0_f32];
    let weights = if morph_weights.is_empty() { &zero_weight[..] } else { &morph_weights };
    let morph_weights_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("phantom morph weights"),
      contents: bytemuck::cast_slice(weights),
      usage: wgpu::BufferUsages::STORAGE,
    });
    let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("phantom skin pose bind"),
      layout: &self.skin_layout,
      entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: palette.as_entire_binding() },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: animated._morph_positions.as_entire_binding(),
        },
        wgpu::BindGroupEntry { binding: 2, resource: morph_weights_buffer.as_entire_binding() },
      ],
    });
    let opacity_data = [opacity.clamp(0.0, 1.0), 0.0, 0.0, 0.0];
    let opacity_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("phantom opacity"),
      contents: bytemuck::cast_slice(&opacity_data),
      usage: wgpu::BufferUsages::UNIFORM,
    });
    let opacity_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("phantom opacity bind"),
      layout: &self.phantom_opacity_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: opacity_buffer.as_entire_binding(),
      }],
    });
    self.phantoms.push(PhantomPose {
      _palette: palette,
      _morph_weights: morph_weights_buffer,
      bind,
      _opacity: opacity_buffer,
      opacity_bind,
    });
  }

  pub fn clear_phantoms(&mut self) {
    self.phantoms.clear();
  }

  /// Advances and uploads one animated frame. The vertex buffer is reused,
  /// so animation does not allocate GPU resources or reset the camera.
  pub fn update_animation(&mut self, delta_seconds: f32) {
    if !self.animation_playing {
      return;
    }
    let Some(index) = self.animation_index else {
      return;
    };
    self.animation_time += delta_seconds.max(0.0) * self.animation_speed;
    self.upload_animation_pose(Some(index), self.animation_time);
    self.update_follow_camera();
  }

  /// Updates the GPU palette only. No animated vertex is transformed or
  /// copied by the CPU; the vertex shader performs the four-weight skinning.
  fn upload_animation_pose(&mut self, animation: Option<usize>, time: f32) {
    let Some(scene) = self.scene.as_ref() else {
      return;
    };
    let Some(animated) = self.animated.as_mut() else {
      return;
    };
    scene.update_gpu_pose(
      animation,
      time,
      &mut animated.transforms,
      &mut animated.morph_weights_cpu,
    );
    self.queue.write_buffer(
      &animated.palette,
      0,
      bytemuck::cast_slice(&animated.transforms),
    );
    self.queue.write_buffer(
      &animated.morph_weights,
      0,
      bytemuck::cast_slice(&animated.morph_weights_cpu),
    );
  }

  /// Computes a conservative radius for the mesh bounds used by the camera's
  /// depth range. Camera poses can move far away from the initial framing, so
  /// using a unit radius here would clip large FBX/glTF assets.
  fn bounds_radius(min: Vec3, max: Vec3) -> f32 {
    ((max - min) * 0.5).length().max(0.01)
  }

  /// Applies a script-provided camera pose and keeps the clipping range tied
  /// to the actual mesh rather than to a placeholder unit-sized object.
  pub fn set_camera(&mut self, c: CameraConfig) {
    self.camera_follow = None;
    self.camera.set_camera(c);
    self
      .camera
      .update_planes(Self::bounds_radius(self.mesh.min, self.mesh.max));
  }

  /// Builds a low-frequency path from the average position of every surface
  /// vertex in the selected animation. Sampling the whole pose once here is
  /// intentional: following the instantaneous AABB center makes the camera
  /// react to individual limb poses, whereas this centroid represents the
  /// animation's overall movement. Runtime target evaluation then only reads
  /// this compact path and performs spline interpolation.
  fn ensure_camera_target_path(&mut self) {
    let Some(scene) = self.scene.as_ref() else {
      return;
    };
    let animation = self.animation_index;
    if self
      .camera_target_path
      .as_ref()
      .is_some_and(|path| path.animation == animation)
    {
      return;
    }

    let duration = animation
      .and_then(|index| scene.animations().get(index).map(|info| info.duration))
      .unwrap_or(0.0)
      .max(0.0);
    // Thirty target samples per second are enough for a smooth Catmull–Rom
    // camera path while keeping the one-time CPU preparation bounded.
    let sample_count = if duration > 0.0 {
      (duration * 30.0).ceil() as usize
    } else {
      0
    };
    let mut targets = Vec::with_capacity(sample_count + 1);
    let mut radius = 0.01_f32;
    for sample in 0..=sample_count {
      let time = if sample == sample_count {
        duration
      } else {
        sample as f32 / 30.0
      };
      let pose = scene.sample(animation, time);
      let mut average = Vec3::ZERO;
      for vertex in &pose.vertices {
        average += Vec3::from_array(vertex.position);
      }
      if !pose.vertices.is_empty() {
        average /= pose.vertices.len() as f32;
      } else {
        average = (pose.min + pose.max) * 0.5;
      }
      // Bound the complete pose around the centroid, not around its AABB
      // center. This remains conservative when the centroid is offset by an
      // asymmetric pose and gives the camera a stable far plane.
      let pose_radius = pose
        .vertices
        .iter()
        .map(|vertex| (Vec3::from_array(vertex.position) - average).length())
        .fold(0.01, f32::max);
      radius = radius.max(pose_radius);
      targets.push(average);
    }

    // Apply a binomial three-point stencil repeatedly. Keeping the endpoints
    // fixed preserves the authored first/last pose, while interior points are
    // averaged with their temporal neighbors on every pass. This is much
    // steadier than asking the camera to follow a freshly deformed pose box.
    for _ in 0..CAMERA_TARGET_STENCIL_PASSES {
      if targets.len() < 3 {
        break;
      }
      let mut smoothed = targets.clone();
      for index in 1..targets.len() - 1 {
        smoothed[index] = (targets[index - 1] + 2.0 * targets[index] + targets[index + 1]) * 0.25;
      }
      targets = smoothed;
    }

    self.camera_target_path = Some(CameraTargetPath {
      animation,
      duration,
      targets,
      radius,
    });
  }

  /// Applies a camera orbit relative to the smoothed animated centroid path.
  /// `update_follow_camera` resolves that path after animation time advances,
  /// so the camera tracks the overall motion rather than individual limbs.
  pub fn set_camera_follow_mesh(&mut self, c: CameraFollowConfig) {
    self.camera_follow = Some(ActiveCameraFollow {
      offset: c.offset,
      up:     c.up,
      fov:    c.fov,
    });
    if !self.animation_playing {
      self.update_follow_camera();
    }
  }

  fn update_follow_camera(&mut self) {
    let Some(follow) = self.camera_follow else {
      return;
    };

    self.ensure_camera_target_path();
    let (target, radius) = if let Some(path) = self.camera_target_path.as_ref() {
      (path.sample(self.animation_time), path.radius)
    } else {
      (
        (self.mesh.min + self.mesh.max) * 0.5,
        Self::bounds_radius(self.mesh.min, self.mesh.max),
      )
    };
    self.camera.set_camera(CameraConfig {
      eye: target + follow.offset,
      target,
      up: follow.up,
      fov: follow.fov,
    });
    self.camera.update_planes(radius);
  }

  /// Sets the linear RGB clear color used for the next rendered frame.
  ///
  /// Scripts pass normalized RGB values. Invalid or out-of-range values are
  /// sanitized here so a malformed JavaScript/Lua value cannot poison the
  /// surface clear operation or produce backend-specific NaNs.
  pub fn set_background_color(&mut self, color: [f32; 3]) {
    self.background_color = color.map(|channel| {
      if channel.is_finite() {
        channel.clamp(0.0, 1.0)
      } else {
        0.0
      }
    });
  }

  /// Hides the scene without releasing the shared device or GPU buffers.
  pub fn hide_scene(&mut self) {
    self.scene_visible = false;
    self.animation_playing = false;
  }

  /// Reveals the current scene after slide setup has completed.
  pub fn show_scene(&mut self) { self.scene_visible = true; }

  /// Captures the next native offscreen frame as a PNG.
  #[cfg(not(target_arch = "wasm32"))]
  pub fn request_screenshot(&mut self, path: impl Into<std::path::PathBuf>) {
    self.screenshot_path = Some(path.into());
    self.screenshot_complete = false;
  }

  /// Returns true after the one-shot screenshot requested by Lua is written.
  #[cfg(not(target_arch = "wasm32"))]
  pub fn screenshot_complete(&self) -> bool { self.screenshot_complete }

  /// Encodes and presents one frame.
  ///
  /// Rendering is demand-driven by winit's `RedrawRequested` event.  Camera
  /// changes therefore become visible only after the app requests a redraw.
  pub fn render(&mut self) {
    let frame = match self.surface.as_ref() {
      Some(surface) => match surface.get_current_texture() {
        Ok(frame) => Some(frame),
        Err(wgpu::SurfaceError::Lost) => {
          // Lost surfaces are recoverable: configure the current size
          // again and let the next redraw try to acquire a frame.
          self.resize(winit::dpi::PhysicalSize::new(
            self.config.width,
            self.config.height,
          ));
          return;
        }
        Err(_) => {
          // Timeout/outdated frames are transient on browsers during
          // resize or tab scheduling.  Dropping this frame avoids a
          // panic; the regular redraw loop will retry.
          return;
        }
      },
      None => None,
    };
    #[cfg(not(target_arch = "wasm32"))]
    let screenshot_path = self.screenshot_path.take();
    let target_texture = frame
      .as_ref()
      .map(|frame| &frame.texture)
      .or_else(|| {
        #[cfg(not(target_arch = "wasm32"))]
        {
          self.headless_target.as_ref()
        }
        #[cfg(target_arch = "wasm32")]
        {
          None
        }
      });
    let Some(target_texture) = target_texture else {
      // A detached browser viewer remains alive so its device can be reused
      // by the next slide, but it must not render until a surface returns.
      return;
    };
    // Keep scripted, interactive, and resize-driven camera changes inside a
    // clipping range derived from the rendered mesh. Follow-camera mode has
    // already installed a pose-specific range after animation updates.
    if self.camera_follow.is_none() {
      self
        .camera
        .update_planes(Self::bounds_radius(self.mesh.min, self.mesh.max));
    }
    let view = target_texture.create_view(&Default::default());
    let m = self.camera.view_projection().to_cols_array_2d();
    // Upload the latest camera before encoding the draw call.  The same
    // buffer is read by both vertex and fragment shader stages.
    self.queue.write_buffer(
      &self.uniform,
      0,
      bytemuck::bytes_of(&Uniforms {
        view_proj:   m,
        camera:      self.camera.eye.extend(1.0).to_array(),
        encode_srgb: [self.encode_srgb, 0.0, 0.0, 0.0],
      }),
    );
    let mut enc = self
      .device
      .create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render"),
      });
    // First render the opaque mesh into an MSAA target and resolve it into the
    // presentation texture. Motion lines are composited in later passes so
    // their transparency can be accumulated independently of draw order.
    {
      let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label:                    Some("mesh pass"),
        color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
          view:           self.msaa_color.as_ref().unwrap_or(&view),
          resolve_target: (self.sample_count > 1).then_some(&view),
          ops:            wgpu::Operations {
            load:  wgpu::LoadOp::Clear(wgpu::Color {
              // On an sRGB target wgpu converts these linear values during
              // presentation. A browser UNORM fallback receives the explicit
              // shader-compatible encoding from clear_channel instead.
              r: clear_channel(self.background_color[0] as f64, self.encode_srgb),
              g: clear_channel(self.background_color[1] as f64, self.encode_srgb),
              b: clear_channel(self.background_color[2] as f64, self.encode_srgb),
              a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
          },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
          view:        &self.depth,
          depth_ops:   Some(wgpu::Operations {
            load:  wgpu::LoadOp::Clear(1.0),
            store: wgpu::StoreOp::Store,
          }),
          stencil_ops: None,
        }),
        occlusion_query_set:      None,
        timestamp_writes:         None,
      });
      pass.set_bind_group(0, &self.bind, &[]);
      if self.scene_visible {
        if let Some(animated) = &self.animated {
          // Animated geometry stays immutable; bind the current pose palette
          // and let the vertex shader perform skinning.
          pass.set_pipeline(&self.animated_pipeline);
          pass.set_bind_group(1, &animated.bind, &[]);
          pass.set_vertex_buffer(0, animated.vertex.slice(..));
          pass.set_index_buffer(animated.index.slice(..), wgpu::IndexFormat::Uint32);
          pass.draw_indexed(0..animated.count, 0, 0..1);
        } else {
          pass.set_pipeline(&self.pipeline);
          pass.set_vertex_buffer(0, self.vertex.slice(..));
          pass.set_index_buffer(self.index.slice(..), wgpu::IndexFormat::Uint32);
          pass.draw_indexed(0..self.count, 0, 0..1);
        }
      }
      if let Some(animated) = &self.animated {
        if !self.phantoms.is_empty() {
          pass.set_pipeline(&self.phantom_pipeline);
          pass.set_vertex_buffer(0, animated.vertex.slice(..));
          pass.set_index_buffer(animated.index.slice(..), wgpu::IndexFormat::Uint32);
          for phantom in &self.phantoms {
            pass.set_bind_group(1, &phantom.bind, &[]);
            pass.set_bind_group(2, &phantom.opacity_bind, &[]);
            pass.draw_indexed(0..animated.count, 0, 0..1);
          }
        }
      }
    }

    if self.scene_visible || (!self.phantoms.is_empty() && self.seed_points_visible) {
      if let Some(lines) = &self.motion_lines {
      // The extracted bundle is complete, but the teaser style displays a
      // moving temporal tail. Updating this tiny uniform is enough to animate
      // the style without rebuilding or copying the trajectories.
      let duration = self
        .animation_index
        .and_then(|index| {
          self
            .scene
            .as_ref()
            .and_then(|scene| scene.animations().get(index).map(|info| info.duration))
        })
        .map(|duration| duration.max(0.0))
        .unwrap_or(0.0);
      let now = if duration > 0.0 {
        self.animation_time.rem_euclid(duration)
      } else {
        0.0
      };
      let model_diagonal = (self.mesh.max - self.mesh.min).length().max(1.0e-4);
      let (tail_duration, strip_width, halo_depth, style_flag) = match self.motion_line_style {
        MotionLineRenderStyle::Teaser => (0.75, 34.0, 0.01, 0.0),
        // Retain the Compasso width construction, but use a two-times longer
        // visible tail so the dash rhythm remains readable in the slides.
        MotionLineRenderStyle::Dashed => (1.6, 8.0, 0.0, 1.0),
        MotionLineRenderStyle::FullTrajectory => (duration, model_diagonal * 0.012, 0.0, 2.0),
        MotionLineRenderStyle::WindowedFullTrajectory =>
          (0.75, model_diagonal * 0.012, 0.0, 2.0),
        MotionLineRenderStyle::UnweightedTeaser => (0.75, 17.0, 0.0, 3.0),
      };
      self.queue.write_buffer(
        &lines.style,
        0,
        bytemuck::bytes_of(&MotionLineStyle {
          timing:   [now, tail_duration, model_diagonal, 0.0],
          widths:   [strip_width, halo_depth, style_flag, self.motion_line_opacity],
          viewport: [
            self.config.width as f32,
            self.config.height as f32,
            0.0,
            0.0,
          ],
        }),
      );

      // Accumulate every line into the two weighted OIT targets. The depth
      // attachment remains the mesh depth, so surface occlusion still works.
      {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
          label:                    Some("motion-line weighted OIT pass"),
          color_attachments:        &[
            Some(wgpu::RenderPassColorAttachment {
              view:           self
                .motion_oit
                .accumulation_msaa_view
                .as_ref()
                .unwrap_or(&self.motion_oit.accumulation_view),
              resolve_target: (self.sample_count > 1).then_some(&self.motion_oit.accumulation_view),
              ops:            wgpu::Operations {
                load:  wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
              },
            }),
            Some(wgpu::RenderPassColorAttachment {
              view:           self
                .motion_oit
                .revealage_msaa_view
                .as_ref()
                .unwrap_or(&self.motion_oit.revealage_view),
              resolve_target: (self.sample_count > 1).then_some(&self.motion_oit.revealage_view),
              ops:            wgpu::Operations {
                load:  wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
              },
            }),
          ],
          depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view:        &self.depth,
            depth_ops:   Some(wgpu::Operations {
              load:  wgpu::LoadOp::Load,
              store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
          }),
          occlusion_query_set:      None,
          timestamp_writes:         None,
        });
        pass.set_bind_group(0, &self.bind, &[]);
        if self.motion_lines_visible {
          pass.set_pipeline(match self.motion_line_style {
            MotionLineRenderStyle::Teaser => &self.motion_line_pipeline,
            MotionLineRenderStyle::Dashed => &self.motion_line_dashed_pipeline,
            MotionLineRenderStyle::FullTrajectory => &self.motion_line_full_trajectory_pipeline,
            MotionLineRenderStyle::WindowedFullTrajectory =>
              &self.motion_line_windowed_full_trajectory_pipeline,
            MotionLineRenderStyle::UnweightedTeaser => &self.motion_line_unweighted_pipeline,
          });
          pass.set_bind_group(1, &lines.line_bind, &[]);
          // Two vertices per trajectory sample form one continuous strip per
          // instance. Invalid fixed-capacity tail samples duplicate the last
          // real sample and are discarded by the fragment stage.
          let strip_vertices = lines.output_stride.saturating_mul(2);
          pass.draw(0..strip_vertices, 0..lines.seed_count);
        }
        if self.seed_points_visible && self.scene_visible {
          if let Some(animated) = &self.animated {
            pass.set_pipeline(&self.seed_point_pipeline);
            pass.set_bind_group(1, &animated.bind, &[]);
            pass.set_bind_group(2, &lines.seed_bind, &[]);
            pass.draw(0..6, 0..lines.seed_count);
          }
        }
        if self.seed_points_visible {
          for phantom in &self.phantoms {
            pass.set_pipeline(&self.seed_point_pipeline);
            pass.set_bind_group(1, &phantom.bind, &[]);
            pass.set_bind_group(2, &lines.seed_bind, &[]);
            pass.draw(0..6, 0..lines.seed_count);
          }
        }
      }

      // Resolve the weighted average and revealage into the presentation
      // target. The composite itself uses ordinary alpha blending only once;
      // the expensive order-dependent line overlap has already been removed.
      {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
          label:                    Some("motion-line OIT composite pass"),
          color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
            view:           &view,
            resolve_target: None,
            ops:            wgpu::Operations {
              load:  wgpu::LoadOp::Load,
              store: wgpu::StoreOp::Store,
            },
          })],
          depth_stencil_attachment: None,
          occlusion_query_set:      None,
          timestamp_writes:         None,
        });
        pass.set_pipeline(&self.motion_composite_pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_bind_group(1, &self.motion_composite_bind, &[]);
        pass.draw(0..3, 0..1);
      }
      }
    }
    #[cfg(not(target_arch = "wasm32"))]
    let screenshot = screenshot_path.map(|path| {
      let width = self.config.width.max(1);
      let height = self.config.height.max(1);
      let unpadded_bytes_per_row = width.saturating_mul(4);
      let bytes_per_row = align_copy_row(unpadded_bytes_per_row);
      let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native screenshot readback"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
      });
      enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
          texture: target_texture,
          mip_level: 0,
          origin: wgpu::Origin3d::ZERO,
          aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
          buffer: &buffer,
          layout: wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(height),
          },
        },
        wgpu::Extent3d {
          width,
          height,
          depth_or_array_layers: 1,
        },
      );
      (path, buffer, width, height, bytes_per_row)
    });

    // Submit commands first, then present the acquired swap-chain frame when
    // this is a surface-backed viewer. Headless previews have no frame to
    // present.
    self.queue.submit(Some(enc.finish()));
    if let Some(frame) = frame {
      frame.present();
    }

    #[cfg(not(target_arch = "wasm32"))]
    if let Some((path, buffer, width, height, bytes_per_row)) = screenshot {
      let slice = buffer.slice(..);
      let (sender, receiver) = std::sync::mpsc::channel();
      slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
      });
      let map_result = self.device.poll(wgpu::PollType::Wait);
      let result = receiver
        .recv()
        .ok()
        .and_then(|result| result.ok().map(|()| map_result));
      match result {
        Some(Ok(_)) => {
          let mapped = slice.get_mapped_range();
          let mut pixels = vec![0_u8; (width as usize) * (height as usize) * 4];
          let bgra = matches!(
            self.config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
          );
          for row in 0..height as usize {
            let source_start = row * bytes_per_row as usize;
            let source = &mapped[source_start..source_start + width as usize * 4];
            let destination_start = row * width as usize * 4;
            for pixel in 0..width as usize {
              let source_pixel = &source[pixel * 4..pixel * 4 + 4];
              let destination = &mut pixels[destination_start + pixel * 4..destination_start + pixel * 4 + 4];
              if bgra {
                destination.copy_from_slice(&[
                  source_pixel[2],
                  source_pixel[1],
                  source_pixel[0],
                  source_pixel[3],
                ]);
              } else {
                destination.copy_from_slice(source_pixel);
              }
            }
          }
          drop(mapped);
          buffer.unmap();
          if let Err(error) = write_png(&path, width, height, &pixels) {
            eprintln!("could not write screenshot {}: {error:#}", path.display());
          }
        }
        Some(Err(error)) => {
          eprintln!("could not map screenshot readback: {error}");
        }
        None => {
          eprintln!("screenshot readback callback did not complete");
        }
      }
      self.screenshot_complete = true;
    }
  }
}

#[cfg(not(target_arch = "wasm32"))]
fn align_copy_row(bytes_per_row: u32) -> u32 {
  let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
  bytes_per_row.div_ceil(alignment) * alignment
}

#[cfg(not(target_arch = "wasm32"))]
fn write_png(path: &std::path::Path, width: u32, height: u32, pixels: &[u8]) -> Result<()> {
  if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
    std::fs::create_dir_all(parent)?;
  }
  let row_size = width as usize * 4;
  let mut raw = Vec::with_capacity((row_size + 1) * height as usize);
  for row in pixels.chunks_exact(row_size).take(height as usize) {
    raw.push(0);
    raw.extend_from_slice(row);
  }
  let mut png = Vec::new();
  png.extend_from_slice(b"\x89PNG\r\n\x1a\n");
  let mut header = Vec::with_capacity(13);
  header.extend_from_slice(&width.to_be_bytes());
  header.extend_from_slice(&height.to_be_bytes());
  header.extend_from_slice(&[8, 6, 0, 0, 0]);
  png.extend_from_slice(&png_chunk(*b"IHDR", &header));
  png.extend_from_slice(&png_chunk(*b"IDAT", &zlib_store(&raw)));
  png.extend_from_slice(&png_chunk(*b"IEND", &[]));
  std::fs::write(path, png)?;
  Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn png_chunk(kind: [u8; 4], data: &[u8]) -> Vec<u8> {
  let mut chunk = Vec::with_capacity(data.len() + 12);
  chunk.extend_from_slice(&(data.len() as u32).to_be_bytes());
  chunk.extend_from_slice(&kind);
  chunk.extend_from_slice(data);
  chunk.extend_from_slice(&crc32(&[&kind, data].concat()).to_be_bytes());
  chunk
}

#[cfg(not(target_arch = "wasm32"))]
fn zlib_store(data: &[u8]) -> Vec<u8> {
  let mut compressed = vec![0x78, 0x01];
  if data.is_empty() {
    compressed.extend_from_slice(&[1, 0, 0xff, 0xff]);
  } else {
    let mut offset = 0;
    while offset < data.len() {
      let length = (data.len() - offset).min(u16::MAX as usize);
      let final_block = offset + length == data.len();
      compressed.push(u8::from(final_block));
      compressed.extend_from_slice(&(length as u16).to_le_bytes());
      compressed.extend_from_slice(&(!(length as u16)).to_le_bytes());
      compressed.extend_from_slice(&data[offset..offset + length]);
      offset += length;
    }
  }
  compressed.extend_from_slice(&adler32(data).to_be_bytes());
  compressed
}

#[cfg(not(target_arch = "wasm32"))]
fn adler32(data: &[u8]) -> u32 {
  let (mut a, mut b) = (1_u32, 0_u32);
  for byte in data {
    a = (a + u32::from(*byte)) % 65_521;
    b = (b + a) % 65_521;
  }
  (b << 16) | a
}

#[cfg(not(target_arch = "wasm32"))]
fn crc32(data: &[u8]) -> u32 {
  let mut crc = u32::MAX;
  for byte in data {
    crc ^= u32::from(*byte);
    for _ in 0..8 {
      crc = if crc & 1 != 0 {
        (crc >> 1) ^ 0xedb8_8320
      } else {
        crc >> 1
      };
    }
  }
  !crc
}

/// Converts one linear clear-color channel when the surface is not sRGB.
///
/// Mesh colors use the same policy in the WGSL fragment shader. Keeping this
/// conversion here avoids the browser's UNORM canvas appearing nearly black
/// even after the mesh lighting has been corrected.
fn clear_channel(linear: f64, encode_srgb: f32) -> f64 {
  if encode_srgb > 0.5 {
    if linear <= 0.0031308 {
      linear * 12.92
    } else {
      1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
  } else {
    linear
  }
}

#[cfg(target_os = "linux")]
fn process_rss_bytes() -> Option<u64> {
  let status = std::fs::read_to_string("/proc/self/status").ok()?;
  let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
  let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
  kib.checked_mul(1024)
}

#[cfg(not(target_os = "linux"))]
fn process_rss_bytes() -> Option<u64> {
  None
}

fn format_bytes(bytes: u64) -> String {
  const KIB: f64 = 1024.0;
  const MIB: f64 = KIB * 1024.0;
  const GIB: f64 = MIB * 1024.0;
  let bytes_f = bytes as f64;
  if bytes_f >= GIB {
    format!("{:.2} GiB", bytes_f / GIB)
  } else if bytes_f >= MIB {
    format!("{:.2} MiB", bytes_f / MIB)
  } else if bytes_f >= KIB {
    format!("{:.2} KiB", bytes_f / KIB)
  } else {
    format!("{} B", bytes)
  }
}

fn storage_layout_entry(
  binding: u32,
  visibility: wgpu::ShaderStages,
  read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
  wgpu::BindGroupLayoutEntry {
    binding,
    visibility,
    ty: wgpu::BindingType::Buffer {
      ty:                 wgpu::BufferBindingType::Storage { read_only },
      has_dynamic_offset: false,
      min_binding_size:   None,
    },
    count: None,
  }
}

fn uniform_layout_entry(
  binding: u32,
  visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
  wgpu::BindGroupLayoutEntry {
    binding,
    visibility,
    ty: wgpu::BindingType::Buffer {
      ty:                 wgpu::BufferBindingType::Uniform,
      has_dynamic_offset: false,
      min_binding_size:   None,
    },
    count: None,
  }
}

fn texture_layout_entry(
  binding: u32,
  visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
  wgpu::BindGroupLayoutEntry {
    binding,
    visibility,
    ty: wgpu::BindingType::Texture {
      sample_type:    wgpu::TextureSampleType::Float { filterable: false },
      view_dimension: wgpu::TextureViewDimension::D2,
      multisampled:   false,
    },
    count: None,
  }
}

/// Selects MSAA only when every attachment participating in a resolved render
/// pass supports it. This keeps the browser path valid on WebGPU adapters with
/// unusual float or revealage format capabilities instead of failing during
/// pipeline or texture validation.
fn choose_msaa_sample_count(adapter: &wgpu::Adapter, surface_format: wgpu::TextureFormat) -> u32 {
  let formats = [
    surface_format,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::R8Unorm,
    wgpu::TextureFormat::Depth24Plus,
  ];
  let supported = formats.iter().all(|format| {
    let flags = adapter.get_texture_format_features(*format).flags;
    flags.sample_count_supported(PREFERRED_MSAA_SAMPLE_COUNT)
      && (*format == wgpu::TextureFormat::Depth24Plus
        || flags.contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE))
  });
  if supported {
    PREFERRED_MSAA_SAMPLE_COUNT
  } else {
    1
  }
}

fn multisample_state(sample_count: u32) -> wgpu::MultisampleState {
  wgpu::MultisampleState {
    count: sample_count,
    mask: !0,
    alpha_to_coverage_enabled: false,
  }
}

/// Creates the color target used for multisampled mesh rendering. The view is
/// kept separately from the acquired surface because swap-chain textures are
/// always single-sampled and can only be used as resolve targets.
fn msaa_color_view(
  device: &wgpu::Device,
  c: &wgpu::SurfaceConfiguration,
  format: wgpu::TextureFormat,
  sample_count: u32,
) -> Option<wgpu::TextureView> {
  if sample_count <= 1 {
    return None;
  }
  Some(
    device
      .create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh MSAA color"),
        size: wgpu::Extent3d {
          width:                 c.width,
          height:                c.height,
          depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
      })
      .create_view(&Default::default()),
  )
}

/// Creates the depth buffer matching the current surface dimensions and the
/// color attachments' sample count.
fn depth_view(
  device: &wgpu::Device,
  c: &wgpu::SurfaceConfiguration,
  sample_count: u32,
) -> wgpu::TextureView {
  device
    .create_texture(&wgpu::TextureDescriptor {
      label: Some("depth"),
      size: wgpu::Extent3d {
        width:                 c.width,
        height:                c.height,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count,
      dimension: wgpu::TextureDimension::D2,
      format: wgpu::TextureFormat::Depth24Plus,
      usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
      view_formats: &[],
    })
    .create_view(&Default::default())
}
