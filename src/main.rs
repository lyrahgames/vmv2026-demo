use demo::common::*;

fn main() -> Result<()> {
  // Native mode receives one Lua script path and turns it into an ordered
  // command list before the winit event loop starts.
  let script = std::env::args()
    .nth(1)
    .ok_or_else(|| anyhow!("usage: cargo run -- scripts/example1.lua"))?;

  let tasks = demo::lua::tasks(&script)?;

  // User events provide the same command transport used by the web build.
  let event_loop =
    winit::event_loop::EventLoop::<demo::application::ApplicationEvent>::with_user_event()
      .build()?;
  let mut app = demo::application::App::new(tasks);
  app.bind_queue(&event_loop);
  event_loop.run_app(&mut app)?;

  Ok(())
}
