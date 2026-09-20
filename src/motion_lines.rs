//! Seed selection and GPU-side motion-line sampling.
//!
//! A motion-line bundle is deliberately kept separate from animation playback:
//! the selected animation is sampled over its complete duration once, while
//! the ordinary mesh can continue to play, pause, or seek independently.  The
//! CPU prepares the animation palettes because the existing scene sampler owns
//! the format-specific FBX/glTF interpolation logic; a compute shader then
//! applies those palettes to the selected source vertices and writes the full
//! trajectory buffer on the GPU.

/// The first extension point for seed selection algorithms.
///
/// Every selector returns indices into the immutable animated vertex buffer.
/// Consequently a seed is always an actual surface vertex, never an
/// interpolated point or an independently invented sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeedSelectionAlgorithm {
  /// Use every vertex in every loaded surface primitive.
  AllVertices,
  /// Select at most `count` distinct vertices with a deterministic
  /// pseudo-random selector. Determinism keeps native and browser showcases
  /// reproducible while still providing a random vertex subset.
  RandomVertices { count: usize },
}

impl SeedSelectionAlgorithm {
  /// Selects valid source-vertex indices for one animated GPU mesh.
  pub fn select(&self, vertex_count: usize) -> Vec<u32> {
    match self {
      Self::AllVertices => (0..vertex_count as u32).collect(),
      Self::RandomVertices { count } => {
        let target = (*count).min(vertex_count);
        if target == 0 {
          return Vec::new();
        }

        // Partial Fisher-Yates without pulling a general-purpose RNG into the
        // native and wasm builds. The xorshift state is local to this
        // selection, so no global randomness or platform-specific API can
        // make the bundle differ between the two frontends.
        let mut candidates: Vec<u32> = (0..vertex_count as u32).collect();
        let mut state = 0x9E37_79B9_u64 ^ vertex_count as u64 ^ target as u64;
        for index in 0..target {
          state ^= state << 13;
          state ^= state >> 7;
          state ^= state << 17;
          let remaining = vertex_count - index;
          let offset = (state as usize) % remaining;
          candidates.swap(index, index + offset);
        }
        candidates.truncate(target);
        candidates
      }
    }
  }
}

/// Configuration for extracting one complete speedline bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct MotionLineConfig {
  /// The stage that chooses source vertices before GPU tracing.
  pub seed_selection:    SeedSelectionAlgorithm,
  /// Uniform samples per second along the selected animation clip.
  pub frames_per_second: f32,
}

impl MotionLineConfig {
  /// Validates user-facing settings before any GPU buffer is allocated.
  pub fn validate(&self) -> crate::common::Result<()> {
    if !self.frames_per_second.is_finite() || self.frames_per_second <= 0.0 {
      crate::common::bail!("motion-line FPS must be finite and greater than zero")
    }
    if let SeedSelectionAlgorithm::RandomVertices { count } = self.seed_selection {
      if count == 0 {
        crate::common::bail!("random motion-line seed count must be greater than zero")
      }
    }
    Ok(())
  }
}

/// A CPU-side description used while preparing GPU storage buffers.
///
/// `palettes` and `morph_weights` are laid out as sample-major arrays. The
/// compute shader uses the strides to locate the pose belonging to one
/// trajectory sample. `duration` is copied into the GPU parameter block so
/// the final sample remains the exact animation endpoint.
pub(crate) struct MotionLineSamples {
  pub seeds:          Vec<u32>,
  pub palettes:       Vec<crate::mesh::SkinTransform>,
  pub morph_weights:  Vec<f32>,
  pub duration:       f32,
  pub sample_count:   u32,
  pub palette_stride: u32,
  pub morph_stride:   u32,
}

/// Builds the sample data for one selected animation.
pub(crate) fn prepare_samples(
  scene: &crate::scene::AnimatedScene,
  animation: usize,
  vertex_count: usize,
  palette_stride: usize,
  morph_stride: usize,
  config: &MotionLineConfig,
) -> crate::common::Result<MotionLineSamples> {
  config.validate()?;
  let info =
    scene.animations().get(animation).cloned().ok_or_else(|| {
      crate::common::anyhow!("motion-line animation index {animation} is invalid")
    })?;
  let seeds = config.seed_selection.select(vertex_count);
  if seeds.is_empty() {
    crate::common::bail!("motion-line seed selection produced no vertices")
  }

  // Include both endpoints. Clamping the final time avoids wrapping back to
  // the first key when duration * FPS is not an integer.
  let sample_count = ((info.duration.max(0.0) * config.frames_per_second).ceil() as usize)
    .saturating_add(1)
    .max(1);
  let morph_stride = morph_stride.max(1);
  let mut palettes = Vec::with_capacity(sample_count.saturating_mul(palette_stride));
  let mut morph_weights = Vec::with_capacity(sample_count.saturating_mul(morph_stride));
  let mut transforms = Vec::new();
  let mut frame_morph_weights = Vec::new();
  for sample in 0..sample_count {
    let time = if sample + 1 == sample_count {
      info.duration.max(0.0)
    } else {
      sample as f32 / config.frames_per_second
    };
    scene.update_gpu_pose(
      Some(animation),
      time,
      &mut transforms,
      &mut frame_morph_weights,
    );
    if transforms.len() != palette_stride {
      crate::common::bail!("motion-line pose changed its transform palette size")
    }
    palettes.extend_from_slice(&transforms);
    if frame_morph_weights.len() < morph_stride {
      frame_morph_weights.resize(morph_stride, 0.0);
    }
    morph_weights.extend_from_slice(&frame_morph_weights[..morph_stride]);
  }

  Ok(MotionLineSamples {
    seeds,
    palettes,
    morph_weights,
    duration: info.duration.max(0.0),
    sample_count: sample_count as u32,
    palette_stride: palette_stride as u32,
    morph_stride: morph_stride as u32,
  })
}

#[cfg(test)]
mod tests {
  use super::SeedSelectionAlgorithm;

  #[test]
  fn all_vertices_are_valid_source_indices() {
    assert_eq!(
      SeedSelectionAlgorithm::AllVertices.select(4),
      vec![0, 1, 2, 3]
    );
  }

  #[test]
  fn random_selection_is_unique_and_bounded() {
    let seeds = SeedSelectionAlgorithm::RandomVertices { count: 20 }.select(5);
    assert_eq!(seeds.len(), 5);
    assert!(seeds.iter().all(|&seed| seed < 5));
    let mut sorted = seeds.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), seeds.len());
  }
}
