//! GPU resource ownership and the actual mesh render pass.
//!
//! `Viewer` intentionally owns the wgpu surface, device, queue, pipeline, and
//! buffers together.  The event-loop layer only changes camera/mesh state and
//! calls [`Viewer::render`] when winit asks for a frame.

use crate::{
  camera::{Camera, CameraConfig},
  common::*,
  mesh::{Mesh, SkinTransform, SkinnedVertex, Vertex},
  scene::{AnimatedGpuMesh, AnimatedScene, AnimationInfo},
};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;
use winit::window::Window;

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
  count:             u32,
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
  // The fragment shader uses the camera position for a simple headlight.
  camera:      [f32; 4],
  // Non-sRGB browser surface formats display linear shader values too dark.
  // A value of one enables shader-side sRGB encoding for that fallback.
  encode_srgb: [f32; 4],
}

/// All GPU state needed to render one mesh into one window/canvas.
pub struct Viewer {
  /// Public because input handling and scripted paths both update it.
  pub camera:        Camera,
  // CPU copy retained for bounds-based camera reset and replacement.
  mesh:              Mesh,
  // A scene is present only for glTF assets. OBJ replacement clears it so
  // stale animation state can never affect a later static mesh.
  scene:             Option<AnimatedScene>,
  animation_index:   Option<usize>,
  animation_time:    f32,
  animation_speed:   f32,
  animation_playing: bool,
  // The instance is retained on wasm so the existing device can create a
  // surface for each newly visible Slidev canvas. Native applications never
  // reattach a surface after startup.
  #[cfg(target_arch = "wasm32")]
  instance:          wgpu::Instance,
  // `'static` is valid because the Arc<Window> passed to create_surface is
  // retained by AppState for at least as long as this surface.
  surface:           Option<wgpu::Surface<'static>>,
  device:            wgpu::Device,
  queue:             wgpu::Queue,
  config:            wgpu::SurfaceConfiguration,
  pipeline:          wgpu::RenderPipeline,
  animated_pipeline: wgpu::RenderPipeline,
  vertex:            wgpu::Buffer,
  index:             wgpu::Buffer,
  count:             u32,
  uniform:           wgpu::Buffer,
  bind:              wgpu::BindGroup,
  skin_layout:       wgpu::BindGroupLayout,
  animated:          Option<AnimatedGpuState>,
  depth:             wgpu::TextureView,
  // This remains constant for a surface configuration, but is written with
  // every frame beside the camera data for a simple, portable uniform ABI.
  encode_srgb:       f32,
}

impl Drop for Viewer {
  fn drop(&mut self) {
    // WebGPU resources are released asynchronously by the browser. A
    // normal Rust drop only removes wgpu's handles; it does not force the
    // browser-side GPUDevice to release all buffers before the next slide
    // creates another device. Explicit destruction is essential for the
    // slide deck, which replaces animated models repeatedly.
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

  /// Creates the wgpu instance, surface, device, pipeline, and mesh buffers.
  ///
  /// This is async because adapter and device requests may cross the browser
  /// WebGPU promise boundary.  The native caller drives it with pollster.
  pub async fn new(window: Arc<Window>, mesh: Mesh) -> Result<Self> {
    #[cfg(target_arch = "wasm32")]
    let size = Self::initial_surface_size(&window);
    #[cfg(not(target_arch = "wasm32"))]
    let size = window.inner_size();
    // The surface is tied to the window/canvas.  Keeping the Window in an
    // Arc lets wgpu own a static surface handle while AppState owns the Arc too.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window)?;
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference:       wgpu::PowerPreference::HighPerformance,
        compatible_surface:     Some(&surface),
        force_fallback_adapter: false,
      })
      .await?;
    // Request only baseline features so the same renderer works on native
    // GPUs, browser WebGPU implementations, and software adapters.
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
    // Prefer an sRGB format for predictable colors, with the first browser
    // supported format as a fallback. A browser can report no usable
    // formats while a canvas is being torn down or when its WebGPU
    // implementation rejects the requested surface. Return that failure
    // to the caller instead of indexing an empty vector and aborting WASM.
    let format = caps
      .formats
      .iter()
      .copied()
      .find(|f| f.is_srgb())
      .or_else(|| caps.formats.first().copied())
      .ok_or_else(|| anyhow!("WebGPU surface reported no supported formats"))?;
    // Native surfaces normally offer an sRGB target, but some browser
    // canvases expose only an UNORM target. The shader needs to encode its
    // linear lighting output itself in the latter case.
    let encode_srgb = (!format.is_srgb()) as u8 as f32;
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
    // The shader is embedded at compile time, keeping the WASM deployment
    // self-contained instead of requiring a second shader fetch.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label:  Some("mesh shader"),
      source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh.wgsl").into()),
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
      multisample:   Default::default(),
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
      multisample:   Default::default(),
      multiview:     None,
      cache:         None,
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
    let depth = depth_view(&device, &config);
    let count = 0;
    let mut s = Self {
      camera,
      mesh,
      scene: None,
      animation_index: None,
      animation_time: 0.0,
      animation_speed: 1.0,
      animation_playing: false,
      #[cfg(target_arch = "wasm32")]
      instance,
      surface: Some(surface),
      device,
      queue,
      config,
      pipeline,
      animated_pipeline,
      vertex,
      index,
      count,
      uniform,
      bind,
      skin_layout,
      animated: None,
      depth,
      encode_srgb,
    };
    s.reset_camera();
    Ok(s)
  }

  /// Detaches the current canvas without destroying the device or pipelines.
  ///
  /// Slidev changes which canvas is visible as it changes slides. Keeping the
  /// device alive and dropping only this surface avoids creating one browser
  /// GPUDevice per slide while still releasing the old canvas attachment
  /// before its window is dropped.
  #[cfg(target_arch = "wasm32")]
  pub fn detach_surface(&mut self) { self.surface.take(); }

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
    self.depth = depth_view(&self.device, &self.config);
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
      self.depth = depth_view(&self.device, &self.config);
    }
  }

  /// Frames the current mesh using its precomputed bounding box.
  pub fn reset_camera(&mut self) { self.camera.reset_for_bounds(self.mesh.min, self.mesh.max) }

  /// Replaces the current static mesh and clears any glTF playback state.
  pub fn mesh_replace(&mut self, mesh: Mesh) {
    self.scene = None;
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
    self.clear_animated_gpu_state();
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
        usage:    wgpu::BufferUsages::VERTEX,
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
    self.upload_animation_pose(Some(index), 0.0);
  }

  /// Starts the selected animation, defaulting to the first animation.
  pub fn play_animation(&mut self) {
    if self.animation_index.is_none() && !self.animation_infos().is_empty() {
      self.animation_index = Some(0);
      self.animation_time = 0.0;
      self.upload_animation_pose(Some(0), 0.0);
    }
    self.animation_playing = self.animation_index.is_some();
  }

  /// Pauses playback while retaining the selected frame and time.
  pub fn pause_animation(&mut self) { self.animation_playing = false; }

  /// Sets playback speed. Negative values play backwards; zero pauses time
  /// progression without changing the selected playing state.
  pub fn set_animation_speed(&mut self, speed: f32) { self.animation_speed = speed; }

  /// Jumps to a time in the selected animation.
  pub fn set_animation_time(&mut self, time: f32) {
    let Some(index) = self.animation_index else {
      return;
    };
    self.animation_time = time;
    self.upload_animation_pose(Some(index), time);
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

  /// Applies a script-provided camera pose.
  pub fn set_camera(&mut self, c: CameraConfig) { self.camera.set_camera(c) }

  /// Encodes and presents one frame.
  ///
  /// Rendering is demand-driven by winit's `RedrawRequested` event.  Camera
  /// changes therefore become visible only after the app requests a redraw.
  pub fn render(&mut self) {
    let Some(surface) = self.surface.as_ref() else {
      // A detached viewer remains alive so its device can be reused by
      // the next slide, but it must not render until a surface returns.
      return;
    };
    let frame = match surface.get_current_texture() {
      Ok(f) => f,
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
    };
    let view = frame.texture.create_view(&Default::default());
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
    {
      let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label:                    Some("pass"),
        color_attachments:        &[Some(wgpu::RenderPassColorAttachment {
          view:           &view,
          resolve_target: None,
          ops:            wgpu::Operations {
            load:  wgpu::LoadOp::Clear(wgpu::Color {
              // On an sRGB target wgpu converts these linear
              // values during presentation. A browser fallback
              // UNORM target does not, so use the equivalent
              // encoded values there to keep the background
              // visually identical to the native viewer.
              r: clear_channel(0.06, self.encode_srgb),
              g: clear_channel(0.07, self.encode_srgb),
              b: clear_channel(0.09, self.encode_srgb),
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
      if let Some(animated) = &self.animated {
        // Animated geometry stays immutable; bind the current pose
        // palette and let the vertex shader perform skinning.
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
    // Submit commands first, then present the acquired swap-chain frame.
    self.queue.submit(Some(enc.finish()));
    frame.present();
  }
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

/// Creates the depth buffer matching the current surface dimensions.
fn depth_view(device: &wgpu::Device, c: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
  device
    .create_texture(&wgpu::TextureDescriptor {
      label:           Some("depth"),
      size:            wgpu::Extent3d {
        width:                 c.width,
        height:                c.height,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count:    1,
      dimension:       wgpu::TextureDimension::D2,
      format:          wgpu::TextureFormat::Depth24Plus,
      usage:           wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING,
      view_formats:    &[],
    })
    .create_view(&Default::default())
}
