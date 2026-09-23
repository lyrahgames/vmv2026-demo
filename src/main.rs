use demo::common::*;
use winit::{dpi::PhysicalSize, window::Window};

fn main() -> Result<()> {
  // Native mode receives one Lua script path and turns it into an ordered
  // command list before the winit event loop starts.
  let mut args = std::env::args().skip(1);
  let first = args.next();
  let headless = first.as_deref() == Some("--headless");
  let script = if headless { args.next() } else { first };
  let script = script.ok_or_else(|| {
    anyhow!("usage: cargo run -- [--headless] scripts/example1.lua")
  })?;

  let tasks = demo::lua::tasks(&script)?;

  if headless {
    demo::application::run_headless(tasks, PhysicalSize::new(960_u32, 540_u32))?;
    return Ok(());
  }

  // User events provide the same command transport used by the web build.
  let event_loop =
    winit::event_loop::EventLoop::<demo::application::ApplicationEvent>::with_user_event()
      .build()?;
  let mut app = demo::application::App::new(tasks);
  app.set_window_attributes(
    Window::default_attributes().with_inner_size(PhysicalSize::new(960_u32, 540_u32)),
  );
  app.bind_queue(&event_loop);
  event_loop.run_app(&mut app)?;

  Ok(())
}
